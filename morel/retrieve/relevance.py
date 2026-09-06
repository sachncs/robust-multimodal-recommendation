"""Modality-availability relevance and mean relevance.

The relevance score r(i, v) is the arithmetic mean of cosine similarities over
modalities jointly observed for both nodes i and v.
"""

from __future__ import annotations

import numpy as np

from morel.core.errors import ShapeError


def _safe_normalize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(normalized, valid)`` where invalid rows are zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return matrix / safe, (norms.flatten() > 0)


def relevance(
    i: int,
    v: int,
    features: dict[str, np.ndarray],
    mask: np.ndarray,
) -> float:
    """Cosine relevance r(i, v) over jointly observed modalities.

    Args
    ----
    i : int
        Query node id.
    v : int
        Candidate node id.
    features : dict[str, np.ndarray]
        Dict mapping modality name to ``(items, dim)`` float32 arrays.
    mask : np.ndarray
        Binary availability mask of shape ``(items, modalities)``.

    Returns
    -------
    float
        Mean cosine similarity in ``[0, 1]``. Returns 0.0 when no modality is
        jointly observed.
    """
    if not features:
        raise ShapeError("features dict is empty")
    if mask.ndim != 2:
        raise ShapeError(f"mask must be 2-D, got {mask.ndim}-D")
    modalities = list(features.keys())
    numerator = 0.0
    denominator = 0.0
    for mod_idx, name in enumerate(modalities):
        if mask[i, mod_idx] <= 0 or mask[v, mod_idx] <= 0:
            continue
        vec_i = features[name][i].astype(np.float64, copy=False)
        vec_v = features[name][v].astype(np.float64, copy=False)
        norm_i = float(np.linalg.norm(vec_i))
        norm_v = float(np.linalg.norm(vec_v))
        if norm_i <= 0 or norm_v <= 0:
            continue
        numerator += float(np.dot(vec_i, vec_v) / (norm_i * norm_v))
        denominator += 1.0
    if denominator == 0:
        return 0.0
    return numerator / denominator


def mean_relevance(
    i: int,
    nodes: np.ndarray,
    features: dict[str, np.ndarray],
    mask: np.ndarray,
) -> float:
    """Mean of r(i, v) over ``nodes``, excluding the query.

    Vectorised: pre-normalises each modality's features once, then for
    each candidate computes the cosine similarity in one matmul and
    averages over jointly observed modalities.

    Args
    ----
    i : int
        Query node id.
    nodes : np.ndarray
        Candidate node ids (1-D).
    features : dict[str, np.ndarray]
        Dict of feature arrays.
    mask : np.ndarray
        Modality availability mask.

    Returns
    -------
    float
        Mean relevance. Returns 0.0 if no candidates.
    """
    if nodes.size == 0:
        return 0.0
    candidates = np.asarray([int(v) for v in nodes if int(v) != i], dtype=np.int64)
    if candidates.size == 0:
        return 0.0
    n_candidates = candidates.size
    modalities = list(features.keys())
    sims = np.zeros(n_candidates, dtype=np.float64)
    denom_per_node = np.zeros(n_candidates, dtype=np.float64)
    for mod_idx, name in enumerate(modalities):
        feats = features[name]
        normalized, valid = _safe_normalize(feats.astype(np.float64, copy=False))
        query_norm = normalized[i] if valid[i] else None
        if query_norm is None:
            continue
        cand_norms = normalized[candidates]
        cand_valid = valid[candidates]
        available = (mask[i, mod_idx] > 0) & (mask[candidates, mod_idx] > 0) & cand_valid
        if not available.any():
            continue
        available_indices = np.where(available)[0]
        sims_per = cand_norms[available_indices] @ query_norm
        sims[available_indices] += sims_per
        denom_per_node[available_indices] += 1
    denom_per_node[denom_per_node == 0] = 1
    final = sims / denom_per_node
    valid = denom_per_node > 0
    if not valid.any():
        return 0.0
    return float(final[valid].mean())


__all__ = ["relevance", "mean_relevance"]
