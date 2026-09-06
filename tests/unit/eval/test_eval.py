"""Tests for morel.eval.ranking, completion, and protocol."""

from __future__ import annotations

import numpy as np
import pytest

from morel.eval import (
    ablation_results,
    explained_variance,
    map_at_k,
    mrr,
    mse,
    ndcg_at_k,
    per_modality_mse,
    precision_at_k,
    recall_at_k,
    robustness_sweep,
)


class Checker:
    """Aggregated test methods for this module."""

    def recall() -> None:
        labels = np.eye(5, 10, dtype=np.float32)
        s = recall_at_k(labels, labels, k=1)
        assert abs(s - 1.0) < 1e-6

    def invalid() -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            recall_at_k(np.zeros((2, 5)), np.zeros((2, 5)), k=0)

    def ndcg() -> None:
        labels = np.eye(5, 10, dtype=np.float32)
        s = ndcg_at_k(labels, labels, k=2)
        assert abs(s - 1.0) < 1e-6

    def k() -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            ndcg_at_k(np.zeros((2, 5)), np.zeros((2, 5)), k=0)

    def precision() -> None:
        s = precision_at_k(np.eye(3, 5), np.eye(3, 5), k=2)
        assert 0.0 <= s <= 1.0

    def mrr() -> None:
        labels = np.eye(3, 5, dtype=np.float32)
        s = mrr(labels, labels)
        assert abs(s - 1.0) < 1e-6

    def at() -> None:
        labels = np.eye(3, 5, dtype=np.float32)
        s = map_at_k(labels, labels, k=3)
        assert 0.0 <= s <= 1.0

    def mse() -> None:
        arr = np.zeros((3, 4), dtype=np.float32)
        assert mse(arr, arr) == 0.0

    def per() -> None:
        a = np.array([[1.0, 2.0]])
        b = np.array([[0.0, 0.0]])
        out = per_modality_mse({"v": a}, {"v": b})
        assert out["v"] == mse(a, b)

    def explained() -> None:
        arr = np.random.default_rng(0).normal(size=(10, 4)).astype(np.float32)
        assert abs(explained_variance(arr, arr) - 1.0) < 1e-5

    def robustness() -> None:
        scores = np.random.default_rng(0).random((5, 10)).astype(np.float32)
        labels = (np.random.default_rng(1).random((5, 10)) > 0.7).astype(np.float32)
        sweep = robustness_sweep(
            {0.1: scores, 0.5: scores, 0.9: scores},
            labels,
            metrics={"recall": lambda s, labels: recall_at_k(s, labels, k=3)},
        )
        assert sweep.ratios == [0.1, 0.5, 0.9]
        assert len(sweep.metrics["recall"]) == 3

    def ablation() -> None:
        s = np.random.default_rng(0).random((3, 5))
        labels = np.eye(3, 5)
        out = ablation_results({"a": s, "b": labels}, labels, metric=lambda x, y: 1.0)
        assert set(out.keys()) == {"a", "b"}
        assert out["a"] == 1.0
