"""BPR loss and strict negative sampler for downstream recommendation."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from morel.core.errors import DataError


def bpr(pos_scores: torch.Tensor, neg_scores: torch.Tensor, *, eps: float = 1e-10) -> torch.Tensor:
    """Bayesian Personalized Ranking loss.

    Args
    ----
    pos_scores : torch.Tensor
        ``(B,)`` scores for positive items.
    neg_scores : torch.Tensor
        ``(B,)`` scores for negative items.

    Returns
    -------
    torch.Tensor
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

    Vectorised sampler: builds the per-user negative pool once and then
    draws ``count`` negatives per user. Strict: never returns a positive
    item. Raises if a user has so many interactions that no negatives exist.

    Args
    ----
    ui : sp.csr_matrix
        User-item interaction CSR matrix.
    count : int
        Number of negatives per user.
    seed : int
        RNG seed.

    Returns
    -------
    np.ndarray
        Array of shape ``(users, count)`` of int64 item ids.
    """
    users, items = ui.shape
    if count <= 0:
        raise DataError(f"count must be positive, got {count}")
    if count > items:
        raise DataError(f"count ({count}) > items ({items})")
    rng = np.random.default_rng(seed)
    positive_dense = np.asarray(ui.toarray(), dtype=bool)
    neg_pool = np.where(
        ~positive_dense, np.broadcast_to(np.arange(items), positive_dense.shape), -1
    )
    pool = [neg_pool[u][neg_pool[u] >= 0] for u in range(users)]
    min_neg = min(len(p) for p in pool)
    if min_neg < count:
        raise DataError(f"at least one user has only {min_neg} negatives available; need {count}")
    out = np.empty((users, count), dtype=np.int64)
    for u in range(users):
        out[u] = rng.choice(pool[u], size=count, replace=False)
    return out


__all__ = ["bpr", "negatives"]
