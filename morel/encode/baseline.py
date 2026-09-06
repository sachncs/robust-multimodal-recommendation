"""GraphEnc Protocol and simple baselines."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn


class GraphEnc(Protocol):
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

    This is the no-transformer ablation: it satisfies the same contract as
    :class:`~morel.encode.transformer.Transformer` — hidden-width output, and
    one embedding per query when handed a padded subgraph batch — so
    ``Pipeline`` can use it wherever the transformer is used.
    """

    def __init__(self, dims: dict[str, int], pe_dim: int, hidden: int) -> None:
        """Build the projection from concatenated modality features plus PE."""
        super().__init__()
        total = sum(dims.values()) + pe_dim
        self.project = nn.Linear(total, hidden)

    def forward(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
        pe: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence: bool | None = None,
    ) -> torch.Tensor:
        """Concatenate masked features and PE, then project to hidden.

        Args:
            features: Per-modality features, ``(B, d_m)`` or ``(B, N, d_m)``.
            mask: Availability mask matching ``features``.
            pe: Positional encoding matching ``features``.
            attention_mask: ``(B, N)`` validity of each subgraph node. Only
                consulted when ``sequence`` is set.
            sequence: When true, the input carries a node axis and the result
                is mean-pooled over the valid nodes of each subgraph.

        Returns
        -------
            ``(B, hidden)`` embeddings.
        """
        parts = [
            features[name] * mask[..., idx : idx + 1] for idx, name in enumerate(features.keys())
        ]
        parts.append(pe)
        projected: torch.Tensor = self.project(torch.cat(parts, dim=-1))
        if not sequence:
            return projected
        if attention_mask is None:
            pooled: torch.Tensor = projected.mean(dim=1)
            return pooled
        weights = attention_mask.unsqueeze(-1).to(projected.dtype)
        return (projected * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


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
        """Concatenate masked features and PE without projection."""
        parts = [features[name] * mask[..., idx : idx + 1] for idx, name in enumerate(self.dims)]
        parts.append(pe)
        return torch.cat(parts, dim=-1)


class Baseline(nn.Module):
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
        """Delegate to the configured inner encoder."""
        encoded: torch.Tensor = self.inner(features, mask, pe)
        return encoded


__all__ = ["Baseline", "GraphEnc", "Identity", "Sum"]
