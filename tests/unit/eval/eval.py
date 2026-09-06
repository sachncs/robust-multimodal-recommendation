"""Tests for morel.eval.ranking, completion, and protocol."""

from __future__ import annotations

import numpy as np
import pytest

from morel.eval import (
    map,
    modal,
    mrr,
    mse,
    ndcg,
    precision,
    recall,
    results,
    sweep,
    variance,
)


class Checker:
    """Aggregated test methods for this module."""

    def perfect(self) -> None:
        labels = np.eye(5, 10, dtype=np.float32)
        s = recall(labels, labels, k=1)
        assert abs(s - 1.0) < 1e-6

    def k(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            recall(np.zeros((2, 5)), np.zeros((2, 5)), k=0)

    def ndcg(self) -> None:
        labels = np.eye(5, 10, dtype=np.float32)
        s = ndcg(labels, labels, k=2)
        assert abs(s - 1.0) < 1e-6

    def invalid(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            ndcg(np.zeros((2, 5)), np.zeros((2, 5)), k=0)

    def shape(self) -> None:
        s = precision(np.eye(3, 5), np.eye(3, 5), k=2)
        assert 0.0 <= s <= 1.0

    def mrr(self) -> None:
        labels = np.eye(3, 5, dtype=np.float32)
        s = mrr(labels, labels)
        assert abs(s - 1.0) < 1e-6

    def range(self) -> None:
        labels = np.eye(3, 5, dtype=np.float32)
        s = map(labels, labels, k=3)
        assert 0.0 <= s <= 1.0

    def zero(self) -> None:
        arr = np.zeros((3, 4), dtype=np.float32)
        assert mse(arr, arr) == 0.0

    def mse(self) -> None:
        a = np.array([[1.0, 2.0]])
        b = np.array([[0.0, 0.0]])
        out = modal({"v": a}, {"v": b})
        assert out["v"] == mse(a, b)

    def one(self) -> None:
        arr = np.random.default_rng(0).normal(size=(10, 4)).astype(np.float32)
        assert abs(variance(arr, arr) - 1.0) < 1e-5

    def sweep(self) -> None:
        scores = np.random.default_rng(0).random((5, 10)).astype(np.float32)
        labels = (np.random.default_rng(1).random((5, 10)) > 0.7).astype(np.float32)
        sweep_result = sweep(
            {0.1: scores, 0.5: scores, 0.9: scores},
            labels,
            metrics={"recall": lambda s, labels: recall(s, labels, k=3)},
        )
        assert sweep_result.ratios == [0.1, 0.5, 0.9]
        assert len(sweep_result.metrics["recall"]) == 3

    def results(self) -> None:
        s = np.random.default_rng(0).random((3, 5))
        labels = np.eye(3, 5)
        out = results({"a": s, "b": labels}, labels, metric=lambda x, y: 1.0)
        assert set(out.keys()) == {"a", "b"}
        assert out["a"] == 1.0
