"""Anchor retrieval: cosine nearest-neighbor over observed modalities."""

from __future__ import annotations

import numpy as np

from morel.core.errors import GraphError, ShapeError
from morel.core.log import get as get_logger

log = get_logger("retrieve.anchor")


def _validate(features: dict[str, np.ndarray], mask: np.ndarray) -> int:
    """Return the number of items after validation."""
    if not features:
        raise GraphError("features dict is empty")
    items_seen = set()
    for arr in features.values():
        if arr.ndim != 2:
            raise ShapeError(f"feature array must be 2-D, got {arr.ndim}-D")
        items_seen.add(arr.shape[0])
    if len(items_seen) != 1:
        raise ShapeError("feature arrays have inconsistent row counts")
    items = next(iter(items_seen))
    if mask.shape[0] != items:
        raise ShapeError(f"mask rows {mask.shape[0]} != feature rows {items}")
    if mask.ndim != 2 or mask.shape[1] != len(features):
        raise ShapeError(f"mask shape {mask.shape} incompatible with {len(features)} modalities")
    return items


def _cosine_topk(query_vec: np.ndarray, candidates: np.ndarray, k: int) -> np.ndarray:
    """Return top-k indices into ``candidates`` by cosine similarity to ``query_vec``.

    Assumes candidates are L2-normalizable. Zero-norm rows are skipped.
    """
    norms = np.linalg.norm(candidates, axis=1)
    valid = norms > 0
    if not valid.any():
        return np.empty(0, dtype=np.int64)
    safe = np.where(valid, norms, 1.0)
    normalized = candidates / safe[:, None]
    q_norm = np.linalg.norm(query_vec)
    if q_norm <= 0:
        return np.empty(0, dtype=np.int64)
    q = query_vec / q_norm
    sims = normalized @ q
    sims[~valid] = -np.inf
    k_eff = min(k, sims.size)
    top = np.argpartition(-sims, k_eff - 1)[:k_eff]
    top = top[np.argsort(-sims[top])]
    return top


def query(
    query_item: int,
    query_modality: str,
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    top: int = 10,
) -> np.ndarray:
    """Top-k cosine neighbors of ``query_item`` in the given modality.

    Args:
        query_item: Index of the query item.
        query_modality: Name of the modality used for retrieval.
        features: Dict of feature arrays.
        mask: Modality availability mask.
        top: Maximum number of anchors to return.

    Returns
    -------
        Array of anchor indices, sorted by descending similarity, excluding
        the query item.
    """
    if query_modality not in features:
        return np.empty(0, dtype=np.int64)
    items = _validate(features, mask)
    if query_item < 0 or query_item >= items:
        raise GraphError(f"query_item {query_item} out of range [0, {items})")
    if top <= 0:
        raise GraphError(f"top must be positive, got {top}")
    mod_idx = list(features.keys()).index(query_modality)
    observed = np.where(mask[:, mod_idx] > 0)[0]
    if observed.size <= 1:
        return np.empty(0, dtype=np.int64)
    candidates = observed[observed != query_item]
    if candidates.size == 0:
        return np.empty(0, dtype=np.int64)
    query_vec = features[query_modality][query_item]
    candidate_features = features[query_modality][candidates]
    top_local = _cosine_topk(query_vec, candidate_features, top)
    return candidates[top_local]


def batch(
    queries: list[int],
    query_modality: str,
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    top: int = 10,
) -> list[np.ndarray]:
    """Anchor retrieval for a batch of query items.

    Returns
    -------
        List of anchor arrays aligned with ``queries``.
    """
    return [query(q, query_modality, features, mask, top=top) for q in queries]


__all__ = ["query", "batch"]
