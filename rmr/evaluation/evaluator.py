"""Evaluator for recommendation performance."""


import numpy as np

from rmr.evaluation.metrics import ndcg_at_k, recall_at_k


class Evaluator:
    """Evaluate recommendation performance across multiple cutoffs.

    This class computes Recall@K and NDCG@K for a configurable list of K
    values.
    """

    def __init__(self, ks: list[int] = None) -> None:
        """Initialize the evaluator.

        Args:
            ks: List of cutoff values to evaluate. Defaults to [10, 20]
                when ``None``.
        """
        if ks is None:
            ks = [10, 20]
        self.ks = ks

    def evaluate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate scores against ground-truth labels.

        Args:
            scores: Predicted relevance scores of shape (n_users, n_items).
            labels: Binary relevance matrix of shape (n_users, n_items).

        Returns:
            Dictionary mapping metric names (e.g. ``"recall@10"``) to
            their computed values.
        """
        results = {}
        for k in self.ks:
            results[f"recall@{k}"] = recall_at_k(scores, labels, k=k)
            results[f"ndcg@{k}"] = ndcg_at_k(scores, labels, k=k)
        return results
