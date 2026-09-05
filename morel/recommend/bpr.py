"""BPR loss and strict negative sampler for downstream recommendation."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from morel.core.errors import DataError


def bpr(pos_scores: torch.Tensor, neg_scores: torch.Tensor, *, eps: float = 1e-10) -> torch.Tensor:
    """Bayesian Personalized Ranking loss.

    Args:
        pos_scores: ``(B,)`` scores for positive items.
        neg_scores: ``(B,)`` scores for negative items.

    Returns:
        Scalar loss.
    """
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + eps).mean()


def negatives(
    ui: sp.csr_matrix,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    """Sample ``count`` negatives per user.

    Strict: never returns a positive item. Raises if a user has so many
    interactions that no negatives exist.

    Args:
        ui: User-item interaction CSR matrix.
        count: Number of negatives per user.
        seed: RNG seed.

    Returns:
        Array of shape ``(users, count)`` of int64 item ids.
    """
    users, items = ui.shape
    if count <= 0:
        raise DataError(f"count must be positive, got {count}")
    if count > items:
        raise DataError(f"count ({count}) > items ({items})")
    rng = np.random.default_rng(seed)
    out = np.empty((users, count), dtype=np.int64)
    all_items = np.arange(items)
    for u in range(users):
        positives = set(ui.indices[ui.indptr[u] : ui.indptr[u + 1]].tolist())
        remaining = np.setdiff1d(all_items, list(positives), assume_unique=False)
        if remaining.size < count:
            raise DataError(
                f"user {u} has {len(positives)} positives; "
                f"only {remaining.size} negatives available, need {count}"
            )
        chosen = rng.choice(remaining, size=count, replace=False)
        out[u] = chosen
    return out


__all__ = ["bpr", "negatives"]
