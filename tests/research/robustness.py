"""Robustness sweep test: run a full eval.robustness_sweep and assert no NaN."""

from __future__ import annotations

import numpy as np

from morel.eval import ndcg_at_k, recall_at_k
from morel.eval.protocol import robustness_sweep


class Checker:
    """Aggregated test methods for this module."""

    def robustness(self) -> None:
        rng = np.random.default_rng(0)
        users, items = 20, 50
        labels = (rng.random((users, items)) > 0.7).astype(np.float32)
        ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        scores_by_ratio = {r: rng.random((users, items)).astype(np.float32) for r in ratios}
        result = robustness_sweep(
            scores_by_ratio,
            labels,
            metrics={
                "recall@10": lambda s, labels: recall_at_k(s, labels, k=10),
                "ndcg@10": lambda s, labels: ndcg_at_k(s, labels, k=10),
            },
        )
        assert len(result.ratios) == len(ratios)
        for metric_values in result.metrics.values():
            assert all(np.isfinite(v) for v in metric_values)

    def sweep(self) -> None:
        rng = np.random.default_rng(0)
        labels = (rng.random((10, 20)) > 0.8).astype(np.float32)
        result = robustness_sweep(
            {}, labels, metrics={"r@5": lambda s, labels: recall_at_k(s, labels, k=5)}
        )
        assert result.ratios == []
        assert result.metrics == {}

    def returns(self) -> None:
        rng = np.random.default_rng(0)
        labels = (rng.random((10, 20)) > 0.8).astype(np.float32)
        scores_by_ratio = {
            0.2: rng.random((10, 20)),
            0.5: rng.random((10, 20)),
            0.8: rng.random((10, 20)),
        }
        result = robustness_sweep(
            scores_by_ratio,
            labels,
            metrics={"recall@5": lambda s, labels: recall_at_k(s, labels, k=5)},
        )
        assert len(result.metrics["recall@5"]) == 3
