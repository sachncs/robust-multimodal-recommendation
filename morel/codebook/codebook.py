"""Codebook: VQ and GumbelCodebook. Code usage and balance loss helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from morel.core.seed import deterministic


class Codebook(nn.Module, ABC):
    """Abstract base class for vector-quantization codebooks.

    Subclasses implement :meth:`forward` to map a hidden representation to
    a tuple ``(quantized, probs)`` where ``quantized`` has the same shape as
    the input and ``probs`` is a routing distribution of shape ``(B, K)``.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self, hidden: torch.Tensor, *, training: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize hidden representations; implemented by subclasses.

        Args:
            hidden: Input tensor of shape ``(B, D)``.
            training: Whether the codebook is in training mode.

        Returns
        -------
            A tuple ``(quantized, probs)`` where ``quantized`` has the
            same shape as the input and ``probs`` has shape ``(B, K)``.
        """


class VQ(Codebook):
    """Vector-quantizing codebook with straight-through gradient."""

    def __init__(
        self, dim: int, size: int, *, commitment: float = 0.25, seed: int | None = None
    ) -> None:
        """Build a vector-quantizing codebook.

        Args:
            dim: Embedding width; must be positive.
            size: Number of codebook entries; must be positive.
            commitment: Commitment-loss weight.
            seed: If given, initialize the codebook under this seed without
                disturbing the caller's global RNG state.
        """
        super().__init__()
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.size = size
        self.commitment = commitment
        with nullcontext() if seed is None else deterministic(seed):
            self.embeddings = nn.Embedding(size, dim)
            nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(
        self, hidden: torch.Tensor, *, training: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize ``hidden`` to the nearest codebook entry.

        Args:
            hidden: ``(B, dim)`` input.
            training: Ignored; VQ is deterministic.

        Returns
        -------
            Tuple of ``(quantized, indices)`` where ``quantized`` has straight-through
            gradient and ``indices`` are the chosen codebook indices.
        """
        del training
        flat = hidden.view(-1, self.dim)
        distances = (
            flat.pow(2).sum(dim=-1, keepdim=True)
            - 2 * flat @ self.embeddings.weight.t()
            + self.embeddings.weight.pow(2).sum(dim=-1)
        )
        indices = distances.argmin(dim=-1)
        quantized = self.embeddings(indices).view_as(hidden)
        quantized_st = hidden + (quantized - hidden).detach()
        one_hot = F.one_hot(indices.view(-1), self.size).float().view(*hidden.shape[:-1], self.size)
        return quantized_st, one_hot


class GumbelCodebook(Codebook):
    """Codebook that uses a Router for the discrete selection.

    Returns ``(quantized, probs)`` where ``probs`` is the routing distribution
    (pre-mask, suitable for usage/balance losses).
    """

    def __init__(self, dim: int, size: int, *, router: nn.Module, seed: int | None = None) -> None:
        """Build a router-driven codebook.

        Args:
            dim: Embedding width.
            size: Number of codebook entries; must be positive.
            router: Module producing the routing distribution.
            seed: If given, initialize the codebook under this seed without
                disturbing the caller's global RNG state.
        """
        super().__init__()
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        self.size = size
        self.router = router
        with nullcontext() if seed is None else deterministic(seed):
            self.codebook = nn.Embedding(size, dim)
            nn.init.xavier_uniform_(self.codebook.weight)

    def forward(
        self, hidden: torch.Tensor, *, training: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize through the router-selected codebook entries.

        Args:
            hidden: ``(B, dim)`` input.
            training: Whether to add Gumbel noise (forwarded to the router).

        Returns
        -------
            Tuple of ``(quantized, probs)``.
        """
        weights = self.router(hidden, training=training)
        probs = weights.probs
        quantized = probs @ self.codebook.weight
        return quantized, probs


class Noop(Codebook):
    """No-op codebook used for ablations; returns the input and a uniform probs."""

    def __init__(self, dim: int, size: int) -> None:
        super().__init__()
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        self.dim = dim
        self.size = size

    def forward(
        self, hidden: torch.Tensor, *, training: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the input unchanged with a uniform routing distribution."""
        del training
        batch_shape = hidden.shape[:-1]
        probs = torch.full(
            (*batch_shape, self.size), 1.0 / self.size, device=hidden.device, dtype=hidden.dtype
        )
        return hidden, probs


def usage(probs: torch.Tensor, *, eps: float = 1e-10) -> torch.Tensor:
    """KL(bar_p || uniform) codebook usage loss.

    Args:
        probs: ``(B, K)`` routing distribution.

    Returns
    -------
        Scalar tensor.
    """
    bar_p = probs.mean(dim=0)
    uniform = torch.full_like(bar_p, 1.0 / bar_p.shape[0])
    return (bar_p * (torch.log(bar_p + eps) - torch.log(uniform + eps))).sum()


def balance(probs: torch.Tensor) -> torch.Tensor:
    """Codebook load-balancing loss: ``K * sum_e bar_p_e^2`` per the paper."""
    bar_p = probs.mean(dim=0)
    return probs.shape[1] * (bar_p**2).sum()


__all__ = ["VQ", "Codebook", "GumbelCodebook", "Noop", "balance", "usage"]
