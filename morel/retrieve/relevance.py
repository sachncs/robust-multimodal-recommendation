"""Modality-availability relevance and mean relevance.

The relevance score r(i, v) is the arithmetic mean of cosine similarities over
modalities jointly observed for both nodes i and v.
"""

from __future__ import annotations

import numpy as np

from morel.core.errors import ShapeError


def relevance(i: int, v: int, features: dict[str, np.ndarray], mask: np.ndarray) -> float:
    """Cosine relevance r(i, v) over jointly observed modalities.

    Args:
        i: Query node id.
        v: Candidate node id.
        features: Dict mapping modality name to ``(items, dim)`` float32 arrays.
        mask: Binary availability mask of shape ``(items, modalities)``.

    Returns
    -------
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
    i: int, nodes: np.ndarray, features: dict[str, np.ndarray], mask: np.ndarray
) -> float:
    """Mean of r(i, v) over ``nodes``, excluding the query.

    Args:
        i: Query node id.
        nodes: Candidate node ids (1-D).
        features: Dict of feature arrays.
        mask: Modality availability mask.

    Returns
    -------
        Mean relevance. Returns 0.0 if no candidates.
    """
    if nodes.size == 0:
        return 0.0
    scores = [relevance(i, int(v), features, mask) for v in nodes if int(v) != i]
    if not scores:
        return 0.0
    return float(np.mean(scores))


__all__ = ["relevance", "mean_relevance"]
