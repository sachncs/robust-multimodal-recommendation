"""Training losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F


class Loss(Protocol):
    """A training loss."""

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        mask: torch.Tensor,
        aux: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return the scalar loss."""
        ...


@dataclass
class Reconstruction(Loss):
    """Masked MSE reconstruction normalized by missing elements per modality.

    Computes ``sum_m mean( (pred - target)^2 over missing positions * dim )``
    so that modalities with different output dimensions contribute on the same
    scale (paper's per-element convention).
    """

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        mask: torch.Tensor,
        aux: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        del aux
        if not predictions:
            return torch.tensor(0.0)
        modalities = list(predictions.keys())
        total = predictions[modalities[0]].new_zeros(())
        for idx, name in enumerate(modalities):
            missing = (1.0 - mask[:, idx]).unsqueeze(-1)
            diff = (predictions[name] - targets[name]) ** 2
            weighted = missing * diff
            denom = missing.sum().clamp(min=1.0) * predictions[name].shape[-1]
            total = total + weighted.sum() / denom
        return total


@dataclass
class BPR(Loss):
    """BPR loss using provided positive/negative scores."""

    pos: torch.Tensor
    neg: torch.Tensor
    eps: float = 1e-10

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        mask: torch.Tensor,
        aux: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        del predictions, targets, mask
        return -torch.log(torch.sigmoid(self.pos - self.neg) + self.eps).mean()


@dataclass
class Composite(Loss):
    """Linear combination of named loss components."""

    components: dict[str, Loss]
    weights: dict[str, float]

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        mask: torch.Tensor,
        aux: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        total = predictions[next(iter(predictions))].new_zeros(())
        for name, loss in self.components.items():
            weight = self.weights.get(name, 1.0)
            total = total + weight * loss.forward(predictions, targets, mask, aux)
        return total


def ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy helper used by codebook-style losses."""
    return F.cross_entropy(logits, target)


__all__ = ["Loss", "Reconstruction", "BPR", "Composite", "ce"]
