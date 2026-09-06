"""Public API for the morel.eval package."""

from morel.eval.completion import explained_variance, mse, per_modality_mse
from morel.eval.protocol import RobustnessResult, ablation_results, robustness_sweep
from morel.eval.ranking import map_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k

__all__ = [
    "RobustnessResult",
    "ablation_results",
    "explained_variance",
    "map_at_k",
    "mrr",
    "mse",
    "ndcg_at_k",
    "per_modality_mse",
    "precision_at_k",
    "recall_at_k",
    "robustness_sweep",
]
