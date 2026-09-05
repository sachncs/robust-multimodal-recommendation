"""Transformer encoder layer (Pre-LN)."""

from __future__ import annotations

import torch
import torch.nn as nn


class Layer(nn.Module):
    """A single Pre-LN transformer block: MHA + FFN with residuals.

    Order: ``x = x + attn(norm1(x))`` then ``x = x + ffn(norm2(x))``.
    """

    def __init__(self, dim: int, heads: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply self-attention then FFN.

        Args:
            hidden: ``(B, S, dim)`` token embeddings.
            attention_mask: Optional ``(B, S)`` bool with True = keep.

        Returns:
            Updated ``(B, S, dim)`` tensor.
        """
        key_padding: torch.Tensor | None = None
        if attention_mask is not None:
            key_padding = ~attention_mask
        attn_out, _ = self.attn(
            hidden, hidden, hidden, key_padding_mask=key_padding, need_weights=False
        )
        hidden = hidden + attn_out
        hidden = self.norm1(hidden)
        ffn_out = self.ffn(hidden)
        hidden = hidden + ffn_out
        hidden = self.norm2(hidden)
        return hidden


__all__ = ["Layer"]
