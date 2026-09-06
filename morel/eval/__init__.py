"""Public API for the morel.eval package."""

from morel.eval.ablation import BASELINE, ablate, conditions
from morel.eval.ablation import KIND as ABLATIONS
from morel.eval.completion import modal, mse, variance
from morel.eval.protocol import Robust, results, sweep
from morel.eval.ranking import map, mrr, ndcg, precision, recall

__all__ = [
    "ABLATIONS",  # KIND dict from morel.eval.ablation,
    "BASELINE",
    "Robust",
    "ablate",
    "conditions",
    "map",
    "modal",
    "mrr",
    "mse",
    "ndcg",
    "precision",
    "recall",
    "results",
    "sweep",
    "variance",
]
