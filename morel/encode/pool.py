"""Sequence pooling strategies."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Scaled dot-product attention pooling over a sequence dim.

    ``weight = softmax(w^T h / sqrt(d))``; output is ``sum weight * h``.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.score = nn.Linear(dim, 1)
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pool a sequence of tokens into one vector per batch item.

        Args:
            hidden: ``(B, S, dim)`` token embeddings.
            mask: Optional ``(B, S)`` bool tensor; True = valid token.

        Returns:
            ``(B, dim)`` pooled embedding.
        """
        if mask is not None:
            scores = self.score(hidden).squeeze(-1)
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = self.score(hidden).squeeze(-1)
        weights = F.softmax(scores * self.scale, dim=1)
        return (weights.unsqueeze(-1) * hidden).sum(dim=1)


class Mean(nn.Module):
    """Mean pool over a sequence dim, masking invalid tokens."""

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pool a sequence of tokens into one vector per batch item."""
        if mask is None:
            return hidden.mean(dim=1)
        denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        masked = hidden * mask.unsqueeze(-1).float()
        return masked.sum(dim=1) / denom


class Token(nn.Module):
    """Select the first token (CLS-like) of every sequence."""

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return hidden[:, 0, :]


class CLS(nn.Module):
    """Alias for Token pooling."""

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return hidden[:, 0, :]


__all__ = ["Attention", "Mean", "Token", "CLS"]
