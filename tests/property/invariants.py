"""Property-based tests using Hypothesis."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from morel.data.mask import bernoulli
from morel.retrieve.acs import compute
from morel.retrieve.bfs import path
from morel.route import Dense, Gumbel, Top


@st.composite
def random(draw: st.DrawFn) -> sp.csr_matrix:
    n = draw(st.integers(min_value=2, max_value=12))
    arr = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        if draw(st.booleans()):
            arr[i, i + 1] = 1
            arr[i + 1, i] = 1
    return sp.csr_matrix(arr)


class Checker:
    """Aggregated test methods for this module."""

    def acs(graph: sp.csr_matrix, anchor: int) -> None:
        if graph.shape[0] == 0:
            return
        anchor = anchor % graph.shape[0]
        sub = compute(graph, [anchor], fallback="empty")
        assert anchor in sub

    def bernoulli(items: int) -> None:
        mask = bernoulli(items, 3, 0.4, seed=0).to_numpy()
        assert (mask.sum(axis=1) >= 1).all()

    def top() -> None:
        r = Top(dim=8, k=10, p=3, tau=0.5)
        torch.manual_seed(0)
        for _ in range(5):
            x = torch.randn(4, 8)
            out = r(x, training=True)
            assert torch.allclose(out.probs.sum(-1), torch.ones(4), atol=1e-5)

    def dense() -> None:
        r = Dense(dim=8, k=10, tau=0.5)
        torch.manual_seed(0)
        for _ in range(5):
            x = torch.randn(4, 8)
            out = r(x, training=True)
            assert torch.allclose(out.probs.sum(-1), torch.ones(4), atol=1e-5)

    def gumbel() -> None:
        r = Gumbel(dim=8, k=10, tau=0.5)
        torch.manual_seed(0)
        for _ in range(5):
            x = torch.randn(4, 8)
            out = r(x, training=True)
            assert torch.allclose(out.probs.sum(-1), torch.ones(4), atol=1e-5)

    def temperature() -> None:
        """Lower temperature sharpens the routing distribution."""
        torch.manual_seed(0)
        x = torch.randn(8, 8)
        soft = Gumbel(dim=8, k=10, tau=2.0)(x, training=False).probs
        sharp = Gumbel(dim=8, k=10, tau=0.1)(x, training=False).probs
        soft_entropy = -(soft * torch.log(soft + 1e-12)).sum(-1).mean()
        sharp_entropy = -(sharp * torch.log(sharp + 1e-12)).sum(-1).mean()
        assert sharp_entropy < soft_entropy

    def bfs(graph: sp.csr_matrix, start: int, end: int) -> None:
        if graph.shape[0] == 0:
            return
        start = start % graph.shape[0]
        end = end % graph.shape[0]
        p = path(graph, start, end)
        assert all(0 <= n < graph.shape[0] for n in p)
