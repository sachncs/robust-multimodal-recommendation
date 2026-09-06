"""Public API for the morel.eval package."""

from morel.eval.ablation import BASELINE, ablate, conditions
from morel.eval.ablation import KIND as ABLATIONS
from morel.eval.completion import variance, mse, per_modality
from morel.eval.protocol import Robust, ablation_results, robustness_sweep
from morel.eval.ranking import map_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k

__all__ = [
    "ABLATIONS",  # KIND dict from morel.eval.ablation,
    "BASELINE",
    "Robust",
    "ablate",
    "ablation_results",
    "conditions",
    "variance",
    "map_at_k",
    "mrr",
    "mse",
    "ndcg_at_k",
    "per_modality",
    "precision_at_k",
    "recall_at_k",
    "robustness_sweep",
]
