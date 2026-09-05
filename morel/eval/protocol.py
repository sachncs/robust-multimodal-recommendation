"""Robustness and ablation evaluation protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class RobustnessResult:
    """Result of a robustness sweep."""

    ratios: list[float]
    metrics: dict[str, list[float]] = field(default_factory=dict)


def robustness_sweep(
    scores_by_ratio: dict[float, np.ndarray],
    labels: np.ndarray,
    *,
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]],
) -> RobustnessResult:
    """Evaluate a metric across a range of mask ratios.

    Args:
        scores_by_ratio: Mapping from mask ratio to score matrix.
        labels: Ground-truth binary labels.
        metrics: Mapping from metric name to scorer.

    Returns:
        RobustnessResult with one entry per ratio per metric.
    """
    if not scores_by_ratio:
        return RobustnessResult(ratios=[])
    ratios = sorted(scores_by_ratio.keys())
    metric_lists: dict[str, list[float]] = {name: [] for name in metrics}
    for ratio in ratios:
        scores = scores_by_ratio[ratio]
        for name, fn in metrics.items():
            metric_lists[name].append(fn(scores, labels))
    return RobustnessResult(ratios=ratios, metrics=metric_lists)


def ablation_results(
    scores_by_condition: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float],
) -> dict[str, float]:
    """Evaluate a single metric across ablation conditions.

    Args:
        scores_by_condition: Mapping from condition name to score matrix.
        labels: Ground-truth binary labels.
        metric: Scorer function.

    Returns:
        Mapping from condition name to metric value.
    """
    return {name: metric(scores, labels) for name, scores in scores_by_condition.items()}


__all__ = ["RobustnessResult", "robustness_sweep", "ablation_results"]
