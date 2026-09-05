"""Joint graph transformer: input embed -> L Pre-LN layers -> attention pool."""

from __future__ import annotations

import torch
import torch.nn as nn

from morel.encode.input import Input
from morel.encode.layer import Layer
from morel.encode.pool import Attention


class Transformer(nn.Module):
    """L-layer graph transformer with query-pool aggregation.

    Accepts either ``(B, d_m)`` per-node inputs (no sequence dim) or
    ``(B, S, d_m)`` sequence inputs (subgraph tokens).
    """

    def __init__(
        self,
        dims: dict[str, int],
        pe_dim: int,
        hidden: int = 128,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.5,
        pool: str = "attention",
    ) -> None:
        super().__init__()
        if layers <= 0:
            raise ValueError(f"layers must be positive, got {layers}")
        if heads <= 0:
            raise ValueError(f"heads must be positive, got {heads}")
        self.input = Input(dims, pe_dim, hidden, dropout)
        self.layers = nn.ModuleList(
            [Layer(hidden, heads, dropout) for _ in range(layers)]
        )
        if pool == "attention":
            self.pool = Attention(hidden)
        elif pool == "mean":
            from morel.encode.pool import Mean

            self.pool = Mean()
        elif pool == "token":
            from morel.encode.pool import Token

            self.pool = Token()
        else:
            raise ValueError(f"unknown pool: {pool!r}")

    def forward(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
        pe: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence: bool | None = None,
    ) -> torch.Tensor:
        """Encode features and return a single embedding per batch item.

        Args:
            features: Dict of feature tensors.
            mask: Availability mask.
            pe: Positional encoding.
            attention_mask: Optional ``(S,)`` or ``(B, S)`` bool mask over the
                sequence dimension.
            sequence: If True, ``features`` represents a sequence
                ``(S, d_m)`` or ``(B, S, d_m)``. If False, ``features`` is
                per-item ``(B, d_m)``. If None, inferred from ``features`` rank.

        Returns:
            ``(B, hidden)`` tensor.
        """
        hidden = self.input(features, mask, pe)
        is_sequence = sequence if sequence is not None else hidden.dim() == 3
        if is_sequence and hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        if not is_sequence:
            hidden = hidden.unsqueeze(1)
            attention_mask = torch.ones(
                (hidden.shape[0], 1), dtype=torch.bool, device=hidden.device
            )
        elif attention_mask is None:
            attention_mask = torch.ones(
                hidden.shape[:2], dtype=torch.bool, device=hidden.device
            )
        elif attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0).expand(hidden.shape[0], -1)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
        return self.pool(hidden, attention_mask)


__all__ = ["Transformer"]
