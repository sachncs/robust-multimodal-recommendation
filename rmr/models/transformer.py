"""Graph transformer modules for GRE-MC.

Provides an input embedding layer that concatenates masked modality features
with positional encodings, a standard transformer encoder layer, and a
query-pooling graph transformer that aggregates contextualized tokens into a
single query embedding per item.
"""


import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """Concatenates modality features (zero-pads missing ones) and projects."""

    def __init__(
        self,
        input_dims: dict[str, int],
        pe_dim: int,
        hidden_dim: int,
        dropout: float = 0.5,
    ) -> None:
        """Initializes the input embedding layer.

        Args:
            input_dims: Mapping from modality name to feature dimension.
            pe_dim: Dimensionality of positional encodings.
            hidden_dim: Target hidden dimension after projection.
            dropout: Dropout probability.
        """
        super().__init__()
        total_in = sum(input_dims.values()) + pe_dim
        self.project = nn.Linear(total_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
        pe: torch.Tensor,
    ) -> torch.Tensor:
        """Projects concatenated (masked features + PE) into hidden space.

        Supports both per-node inputs (``(B, d_m)``) and batched sequences
        (``(B, S, d_m)``).  The ``mask`` tensor should be either
        ``(B, M)`` or ``(B, S, M)``.

        Args:
            features: Dict of tensors.  Each tensor has shape ``(B, d_m)``
                or ``(B, S, d_m)``.
            mask: Binary indicator matrix of shape ``(B, M)`` or
                ``(B, S, M)``.  A value of 0 means the modality is missing.
            pe: Positional encoding tensor of shape ``(B, pe_dim)`` or
                ``(B, S, pe_dim)``.

        Returns:
            Projected embeddings of shape ``(B, hidden_dim)`` or
            ``(B, S, hidden_dim)``.
        """
        mods = list(features.keys())
        feats_list = []
        for i, m in enumerate(mods):
            feat = features[m]
            mask_slice = mask[..., i : i + 1]
            feat = feat * mask_slice
            feats_list.append(feat)
        x = torch.cat(feats_list + [pe], dim=-1)
        h = self.project(x)
        h = self.dropout(h)
        h = self.norm(h)
        return h


class GraphTransformerLayer(nn.Module):
    """Single transformer layer with multi-head self-attention and FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.5,
    ) -> None:
        """Initializes the transformer layer.

        Args:
            hidden_dim: Hidden dimensionality.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Applies self-attention and feed-forward network with residuals.

        Args:
            x: Input tensor of shape ``(B, seq_len, hidden_dim)``.
            attn_mask: Optional attention mask.

        Returns:
            Output tensor of shape ``(B, seq_len, hidden_dim)``.
        """
        attn_out, _ = self.attn(
            x, x, x, attn_mask=attn_mask, need_weights=False
        )
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class JointEncodingGraphTransformer(nn.Module):
    """L-layer graph transformer with input embedding and query pooling."""

    def __init__(
        self,
        input_dims: dict[str, int],
        pe_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.5,
    ) -> None:
        """Initializes the joint-encoding graph transformer.

        Args:
            input_dims: Mapping from modality name to feature dimension.
            pe_dim: Dimensionality of positional encodings.
            hidden_dim: Hidden dimensionality.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.input_embed = InputEmbedding(
            input_dims, pe_dim, hidden_dim, dropout
        )
        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.query_pool = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
        pe: torch.Tensor,
    ) -> torch.Tensor:
        """Computes query embeddings via transformer and attention pooling.

        Accepts either a single token per batch item (legacy query-only
        mode) or a sequence of tokens per batch item (subgraph encoding).

        Args:
            features: Dict of tensors.  Shapes are ``(B, d_m)`` for
                single-token inputs or ``(B, S, d_m)`` for sequences.
            mask: Binary indicator matrix of shape ``(B, M)`` or
                ``(B, S, M)``.
            pe: Positional encoding tensor of shape ``(B, pe_dim)`` or
                ``(B, S, pe_dim)``.

        Returns:
            Query embeddings of shape ``(B, hidden_dim)``.
        """
        h = self.input_embed(features, mask, pe)
        # If input is per-node (no sequence dimension), add one.
        if h.dim() == 2:
            h = h.unsqueeze(1)  # (B, 1, hidden_dim)
        for layer in self.layers:
            h = layer(h)
        # Attention-weighted aggregation over the sequence dimension.
        w = torch.softmax(
            self.query_pool(h), dim=1
        )  # (B, seq_len, 1)
        z = (w * h).sum(dim=1)  # (B, hidden_dim)
        return z
