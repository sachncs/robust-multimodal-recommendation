"""GraphEncoder Protocol and simple baselines."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn


class GraphEncoder(Protocol):
    """Graph encoder turns modality features into a hidden embedding."""

    def forward(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
        pe: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``(B, hidden)`` embeddings."""
        ...


class Identity(nn.Module):
    """A trivial encoder that just projects concatenated features with a Linear.

    Useful as an ablation baseline.
    """

    def __init__(self, dims: dict[str, int], pe_dim: int, hidden: int) -> None:
        super().__init__()
        total = sum(dims.values()) + pe_dim
        self.project = nn.Linear(total, hidden)

    def forward(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
        pe: torch.Tensor,
    ) -> torch.Tensor:
        parts = [
            features[name] * mask[..., idx : idx + 1] for idx, name in enumerate(features.keys())
        ]
        parts.append(pe)
        return self.project(torch.cat(parts, dim=-1))


class Sum(nn.Module):
    """A summation encoder (no learnable projection)."""

    def __init__(self, dims: dict[str, int], pe_dim: int, hidden: int) -> None:
        super().__init__()
        self.dims = dims
        self.pe_dim = pe_dim
        self.dim = hidden

    def forward(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
        pe: torch.Tensor,
    ) -> torch.Tensor:
        parts = [features[name] * mask[..., idx : idx + 1] for idx, name in enumerate(self.dims)]
        parts.append(pe)
        return torch.cat(parts, dim=-1)


class GraphEncoderBaseline(nn.Module):
    """Multiplexer that builds the requested graph encoder."""

    def __init__(
        self,
        kind: str,
        dims: dict[str, int],
        pe_dim: int,
        hidden: int,
    ) -> None:
        super().__init__()
        if kind == "identity":
            self.inner: nn.Module = Identity(dims, pe_dim, hidden)
        elif kind == "sum":
            self.inner = Sum(dims, pe_dim, hidden)
        else:
            raise ValueError(f"unknown baseline kind: {kind!r}")

    def forward(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
        pe: torch.Tensor,
    ) -> torch.Tensor:
        return self.inner(features, mask, pe)


__all__ = ["GraphEncoder", "Identity", "Sum", "GraphEncoderBaseline"]
