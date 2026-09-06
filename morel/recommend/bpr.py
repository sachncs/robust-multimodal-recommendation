"""BPR loss and strict negative sampler for downstream recommendation."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from morel.core.errors import Datum


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


def distinct(rng: np.random.Generator, high: int, size: int) -> np.ndarray:
    """Draw ``size`` distinct integers from ``[0, high)``.

    Uses rejection sampling when the draw is sparse relative to the range, so
    the cost does not scale with ``high``. Falls back to a full permutation
    draw only when ``size`` is a large fraction of ``high``, where the dense
    approach is both correct and cheap.

    The order of the returned values follows the order they were drawn, which
    keeps the result reproducible for a given generator state.
    """
    if size * 3 >= high:
        return np.asarray(rng.choice(high, size=size, replace=False), dtype=np.int64)
    picked: list[int] = []
    seen: set[int] = set()
    while len(picked) < size:
        for value in rng.integers(0, high, size=(size - len(picked)) * 2 + 8):
            item = int(value)
            if item not in seen:
                seen.add(item)
                picked.append(item)
                if len(picked) == size:
                    break
    return np.asarray(picked, dtype=np.int64)


def to_ranks(ranks: np.ndarray, positives: np.ndarray) -> np.ndarray:
    """Map ranks over the *negative* items back to actual item ids.

    ``ranks`` index the sorted sequence of items that are **not** in
    ``positives``. Shifting each rank by the number of positives that precede
    it recovers the item id without ever materializing the negative pool.

    Args:
        ranks: Ranks in ``[0, items - len(positives))``.
        positives: Sorted, unique positive item ids for one user.

    Returns
    -------
        Item ids of the same shape as ``ranks``.
    """
    if positives.size == 0:
        return ranks
    shifted = positives - np.arange(positives.size, dtype=positives.dtype)
    return ranks + np.searchsorted(shifted, ranks, side="right")


def negatives(
    ui: sp.csr_matrix,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    """Sample ``count`` distinct negatives per user.

    Works directly off the CSR structure: memory is ``O(nnz + users * count)``
    rather than ``O(users * items)``. The previous implementation densified the
    interaction matrix and then built an ``int64`` pool of the same shape,
    which made the sampler unusable on realistically sized catalogues (a
    100k x 50k dataset needed tens of gigabytes).

    Strict: never returns a positive item, and the ``count`` negatives drawn
    for a user are distinct from each other.

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

    Raises
    ------
    Datum
        If ``count`` is not positive, exceeds the catalogue size, or if some
        user has fewer than ``count`` negatives available.
    """
    users, items = ui.shape
    if count <= 0:
        raise Datum(f"count must be positive, got {count}")
    if count > items:
        raise Datum(f"count ({count}) > items ({items})")
    rng = np.random.default_rng(seed)
    indptr = ui.indptr
    indices = ui.indices

    per_user = [np.unique(indices[indptr[u] : indptr[u + 1]]) for u in range(users)]
    min_neg = min(items - positives.size for positives in per_user) if users else items
    if min_neg < count:
        raise Datum(f"at least one user has only {min_neg} negatives available; need {count}")

    out = np.empty((users, count), dtype=np.int64)
    for u, positives in enumerate(per_user):
        ranks = distinct(rng, items - positives.size, count)
        out[u] = to_ranks(ranks, positives)
    return out


__all__ = ["bpr", "distinct", "negatives", "to_ranks"]
