"""Codebook: VQ and GumbelVQ. Code usage and balance loss helpers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQ(nn.Module):
    """Vector-quantizing codebook with straight-through gradient."""

    def __init__(self, dim: int, size: int, *, commitment: float = 0.25) -> None:
        super().__init__()
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.size = size
        self.commitment = commitment
        self.embeddings = nn.Embedding(size, dim)
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize ``hidden`` to the nearest codebook entry.

        Args:
            hidden: ``(B, dim)`` input.

        Returns:
            Tuple of ``(quantized, indices)`` where ``quantized`` has straight-through
            gradient and ``indices`` are the chosen codebook indices.
        """
        flat = hidden.view(-1, self.dim)
        distances = (
            flat.pow(2).sum(dim=-1, keepdim=True)
            - 2 * flat @ self.embeddings.weight.t()
            + self.embeddings.weight.pow(2).sum(dim=-1)
        )
        indices = distances.argmin(dim=-1)
        quantized = self.embeddings(indices).view_as(hidden)
        commitment_loss = F.mse_loss(hidden, quantized.detach())
        codebook_loss = F.mse_loss(quantized, hidden.detach())
        quantized_st = hidden + (quantized - hidden).detach()
        return quantized_st, indices.view(hidden.shape[:-1])


class GumbelVQ(nn.Module):
    """Codebook that uses a Router for the discrete selection.

    Returns ``(quantized, probs)`` where ``probs`` is the routing distribution
    (pre-mask, suitable for usage/balance losses).
    """

    def __init__(self, dim: int, size: int, *, router: nn.Module) -> None:
        super().__init__()
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        self.size = size
        self.router = router
        self.codebook = nn.Embedding(size, dim)
        nn.init.xavier_uniform_(self.codebook.weight)

    def forward(
        self, hidden: torch.Tensor, *, training: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize through the router-selected codebook entries.

        Args:
            hidden: ``(B, dim)`` input.
            training: Whether to add Gumbel noise (forwarded to the router).

        Returns:
            Tuple of ``(quantized, probs)``.
        """
        weights = self.router(hidden, training=training)
        probs = weights.probs
        quantized = probs @ self.codebook.weight
        return quantized, probs


def usage(probs: torch.Tensor, *, eps: float = 1e-10) -> torch.Tensor:
    """KL(bar_p || uniform) codebook usage loss.

    Args:
        probs: ``(B, K)`` routing distribution.

    Returns:
        Scalar tensor.
    """
    bar_p = probs.mean(dim=0)
    uniform = torch.full_like(bar_p, 1.0 / bar_p.shape[0])
    return (bar_p * (torch.log(bar_p + eps) - torch.log(uniform + eps))).sum()


def balance(probs: torch.Tensor) -> torch.Tensor:
    """Codebook load-balancing loss: ``K * sum_e bar_p_e^2`` per the paper."""
    bar_p = probs.mean(dim=0)
    return probs.shape[1] * (bar_p**2).sum()


__all__ = ["VQ", "GumbelVQ", "usage", "balance"]
