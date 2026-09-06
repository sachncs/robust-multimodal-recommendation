"""Router abstractions and concrete implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from morel.core.errors import ConfigError


@dataclass(frozen=True)
class Weights:
    """Routing distribution and any associated metadata."""

    probs: torch.Tensor  # (B, K) nonnegative, sums to 1
    logits: torch.Tensor | None = None

    @property
    def shape(self) -> torch.Size:
        """Shape of the routing distribution."""
        return self.probs.shape


class Router(nn.Module, ABC):
    """Protocol-style base for routers (kept as nn.Module so parameters register)."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self, hidden: torch.Tensor, *, training: bool = True
    ) -> Weights:
        """Route hidden states to a routing distribution.

        Args:
            hidden: Input tensor of shape ``(B, D)``.
            training: Whether the router is in training mode.

        Returns
        -------
            A :class:`Weights` carrying the routing distribution.
        """


class Dense(Router):
    """Plain softmax routing over K entries."""

    def __init__(self, dim: int, k: int, *, tau: float = 1.0) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        self.tau = tau
        self.k = k
        self.linear = nn.Linear(dim, k)

    def forward(self, hidden: torch.Tensor, *, training: bool = True) -> Weights:
        """Route ``hidden`` through the dense softmax layer."""
        logits = self.linear(hidden) / self.tau
        probs = F.softmax(logits, dim=-1)
        return Weights(probs=probs, logits=logits)


class Top(Router):
    """Top-P (top-K) sparsifying router.

    Applies softmax with optional Gumbel noise, then keeps the top-p entries
    and renormalises.
    """

    def __init__(self, dim: int, k: int, *, p: int, tau: float = 0.5) -> None:
        super().__init__()
        if p <= 0:
            raise ValueError(f"p must be positive, got {p}")
        if p > k:
            raise ValueError(f"p ({p}) must be <= k ({k})")
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        self.tau = tau
        self.k = k
        self.p = p
        self.linear = nn.Linear(dim, k)
        self.eps = 1e-10

    def forward(self, hidden: torch.Tensor, *, training: bool = True) -> Weights:
        """Route through softmax and sparsify to the top-p entries."""
        logits = self.linear(hidden) / self.tau
        if training:
            gumbel = -torch.log(-torch.log(torch.rand_like(logits) + self.eps) + self.eps)
            logits = logits + gumbel
        probs = F.softmax(logits, dim=-1)
        if self.p < self.k:
            _, top_idx = torch.topk(probs, self.p, dim=-1)
            mask = torch.zeros_like(probs)
            mask.scatter_(1, top_idx, 1.0)
            masked = probs * mask
            denom = masked.sum(dim=-1, keepdim=True).clamp(min=self.eps)
            probs = masked / denom
        return Weights(probs=probs, logits=logits)


class Fixed(Router):
    """A non-trainable router that returns a uniform distribution.

    The fixed router has no learnable parameters. Without an explicit
    index input it falls back to a uniform distribution over the
    ``k`` codebook entries, which is the maximum-entropy default and
    makes the downstream codebook's behaviour visible end-to-end.
    """

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def forward(self, hidden: torch.Tensor, *, training: bool = True) -> Weights:
        """Return a uniform distribution over ``k`` entries.

        Args:
            hidden: Input tensor of shape ``(B, D)``. Only the batch
                dimension is used.
            training: Unused; the fixed router has no stochastic path.

        Returns
        -------
            A :class:`Weights` carrying a uniform probability over the
            ``k`` codebook entries and zero logits.
        """
        batch = int(hidden.shape[0])
        logits = torch.zeros(batch, self.k, device=hidden.device, dtype=hidden.dtype)
        probs = torch.full(
            (batch, self.k), 1.0 / self.k, device=hidden.device, dtype=hidden.dtype
        )
        return Weights(probs=probs, logits=logits)


class Gumbel(Router):
    """Pure Gumbel-Softmax router (no top-k sparsification)."""

    def __init__(self, dim: int, k: int, *, tau: float = 0.5) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        self.tau = tau
        self.k = k
        self.linear = nn.Linear(dim, k)
        self.eps = 1e-10

    def forward(self, hidden: torch.Tensor, *, training: bool = True) -> Weights:
        """Route through a Gumbel-Softmax over K entries."""
        logits = self.linear(hidden) / self.tau
        if training:
            gumbel = -torch.log(-torch.log(torch.rand_like(logits) + self.eps) + self.eps)
            logits = logits + gumbel
        probs = F.softmax(logits, dim=-1)
        return Weights(probs=probs, logits=logits)


def build(kind: str, dim: int, k: int, *, p: int, tau: float) -> Router:
    """Build a router by name."""
    if kind == "dense":
        return Dense(dim, k, tau=tau)
    if kind == "top":
        return Top(dim, k, p=p, tau=tau)
    if kind == "gumbel":
        return Gumbel(dim, k, tau=tau)
    if kind == "fixed":
        return Fixed(k)
    raise ConfigError(f"unknown router '{kind}'; available: top, dense, gumbel, fixed")


__all__ = ["Dense", "Fixed", "Gumbel", "Router", "Top", "Weights", "build"]
