"""Research validation tests against the GRE-MC paper's claims."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from morel.codebook import balance, usage
from morel.eval import ndcg_at_k, recall_at_k
from morel.recommend import Light
from morel.retrieve.acs import compute


class Checker:
    """Aggregated test methods for this module."""

    def acs(self) -> None:
        """ACS(0, 4) on a 5-node path should be the full path."""
        g = sp.csr_matrix(
            np.array(
                [
                    [0, 1, 0, 0, 0],
                    [1, 0, 1, 0, 0],
                    [0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 1],
                    [0, 0, 0, 1, 0],
                ],
                dtype=np.float32,
            )
        )
        sub = compute(g, [0, 4])
        assert sub == {0, 1, 2, 3, 4}

    def fallback(self) -> None:
        g = sp.csr_matrix(
            np.array(
                [
                    [0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 1],
                    [0, 0, 0, 1, 0],
                ],
                dtype=np.float32,
            )
        )
        sub = compute(g, [0, 2], fallback="anchors")
        assert sub == {0, 2}

    def load(self) -> None:
        """Per the paper, L_load = K * sum_e bar_g_e^2."""
        probs = torch.full((16, 10), 0.1)
        assert abs(float(balance(probs)) - 1.0) < 1e-3

    def usage(self) -> None:
        probs = torch.full((64, 16), 1.0 / 16)
        assert float(usage(probs)) < 1e-5

    def light(self) -> None:
        """L=0 reduces LightGCN to a dot product."""
        users, items = 5, 7
        light = Light(users=users, items=items, embed=8, layers=0)
        ui = sp.csr_matrix(np.eye(users, items, dtype=np.float32))
        light(torch.arange(users), torch.arange(items), ui)
        out = light(torch.arange(users), torch.arange(items))
        expected = light.user_emb.weight @ light.item_emb.weight.t()
        assert torch.allclose(out, expected)

    def recall(self) -> None:
        """Recall@K is bounded in [0, 1]."""
        rng = np.random.default_rng(0)
        labels = (rng.random((10, 20)) > 0.8).astype(np.float32)
        out = recall_at_k(labels, labels, k=5)
        assert 0.0 <= out <= 1.0

    def ndcg(self) -> None:
        rng = np.random.default_rng(0)
        labels = (rng.random((10, 20)) > 0.8).astype(np.float32)
        out = ndcg_at_k(labels, labels, k=5)
        assert 0.0 <= out <= 1.0

    def at(self) -> None:
        """Identity scores with identity labels should give 1.0 (perfect ranking)."""
        labels = np.eye(5, 10, dtype=np.float32)
        out = recall_at_k(labels, labels, k=1)
        assert abs(out - 1.0) < 1e-6