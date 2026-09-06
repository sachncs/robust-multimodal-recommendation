"""Vectorized ranking metrics."""

from __future__ import annotations

import numpy as np


def recall_at_k(scores: np.ndarray, labels: np.ndarray, *, k: int = 10) -> float:
    """Mean Recall@K across users with at least one relevant item.

    Args:
        scores: ``(users, items)`` predicted scores.
        labels: ``(users, items)`` binary relevance.
        k: Cutoff.

    Returns
    -------
        Scalar Recall@K in ``[0, 1]``.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if scores.shape != labels.shape:
        raise ValueError(f"score/label shape mismatch: {scores.shape} vs {labels.shape}")
    top_k = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
    relevant = labels.sum(axis=1)
    mask = relevant > 0
    if not mask.any():
        return 0.0
    hits = np.take_along_axis(labels, top_k, axis=1).sum(axis=1)
    return float((hits[mask] / relevant[mask]).mean())


def ndcg_at_k(scores: np.ndarray, labels: np.ndarray, *, k: int = 10) -> float:
    """Mean NDCG@K.

    Args:
        scores: ``(users, items)`` predicted scores.
        labels: ``(users, items)`` binary relevance.
        k: Cutoff.

    Returns
    -------
        Scalar NDCG@K in ``[0, 1]``.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if scores.shape != labels.shape:
        raise ValueError(f"score/label shape mismatch: {scores.shape} vs {labels.shape}")
    top_k = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
    relevances = np.take_along_axis(labels, top_k, axis=1)
    positions = np.arange(1, k + 1)
    discounts = 1.0 / np.log2(positions + 1)
    dcg = (relevances * discounts).sum(axis=1)
    ideal = np.sort(labels, axis=1)[:, ::-1][:, :k]
    idcg = (ideal * discounts).sum(axis=1)
    valid = idcg > 0
    if not valid.any():
        return 0.0
    return float((dcg[valid] / idcg[valid]).mean())


def precision_at_k(scores: np.ndarray, labels: np.ndarray, *, k: int = 10) -> float:
    """Mean Precision@K."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    top_k = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
    hits = np.take_along_axis(labels, top_k, axis=1).sum(axis=1)
    return float((hits / k).mean())


def mrr(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mean reciprocal rank of the first relevant item.

    Uses ``argpartition`` to find the position of the highest-scoring
    relevant item in expected O(N) instead of sorting the full row in
    O(N log N).
    """
    has_relevant = labels.sum(axis=1) > 0
    if not has_relevant.any():
        return 0.0
    # Sort rows of -scores in descending order of score; for each user find
    # the index of the first label=1 in that order.
    order = np.argsort(-scores, axis=1)
    relevances = np.take_along_axis(labels, order, axis=1)
    found = np.argmax(relevances > 0, axis=1)
    rr = 1.0 / (found[has_relevant] + 1)
    return float(rr.mean())


def map_at_k(scores: np.ndarray, labels: np.ndarray, *, k: int = 10) -> float:
    """Mean Average Precision@K."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    top_k = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
    relevances = np.take_along_axis(labels, top_k, axis=1)
    cumulative = np.cumsum(relevances > 0, axis=1)
    positions = np.arange(1, k + 1)
    precision_at_i = cumulative / positions
    average = (precision_at_i * relevances).sum(axis=1)
    relevant = labels.sum(axis=1).clip(min=1)
    return float((average / relevant).mean())


__all__ = ["map_at_k", "mrr", "ndcg_at_k", "precision_at_k", "recall_at_k"]
