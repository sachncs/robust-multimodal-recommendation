"""Input embedding: concatenate masked modality features with PE and project."""

from __future__ import annotations

import torch
import torch.nn as nn


class Input(nn.Module):
    """Concatenate per-modality features (zero-pad missing) with PE, then project.

    ``mask`` semantics: 1 = kept (feature present), 0 = missing (feature zeroed).
    """

    def __init__(
        self, dims: dict[str, int], pe_dim: int, hidden: int, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.modalities = list(dims.keys())
        total = sum(dims.values()) + pe_dim
        self.project = nn.Linear(total, hidden)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
        pe: torch.Tensor,
    ) -> torch.Tensor:
        """Project concatenated input.

        Args:
            features: Dict mapping modality name to ``(B, d_m)`` or
                ``(B, S, d_m)`` tensor.
            mask: ``(B, M)`` or ``(B, S, M)`` availability mask.
            pe: ``(B, pe_dim)`` or ``(B, S, pe_dim)`` positional encoding.

        Returns
        -------
            ``(B, hidden)`` or ``(B, S, hidden)`` projected tensor.
        """
        parts: list[torch.Tensor] = []
        for idx, name in enumerate(self.modalities):
            feat = features[name]
            slice_ = mask[..., idx : idx + 1]
            feat = feat * slice_
            parts.append(feat)
        parts.append(pe)
        x = torch.cat(parts, dim=-1)
        x = self.project(x)
        x = self.dropout(x)
        normalized: torch.Tensor = self.norm(x)
        return normalized


__all__ = ["Input"]
