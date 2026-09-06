"""Tests for morel.retrieve.mage."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from morel.core.errors import Net
from morel.retrieve.mage import expand


@pytest.fixture
def setup() -> tuple[dict[str, np.ndarray], np.ndarray, sp.csr_matrix]:
    g = sp.csr_matrix(
        np.array(
            [[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0]],
            dtype=np.float32,
        )
    )
    features = {"visual": np.eye(5, dtype=np.float32), "text": np.eye(5, dtype=np.float32)}
    mask = np.ones((5, 2), dtype=np.float32)
    return features, mask, g


class Checker:
    """Aggregated test methods for this module."""

    def query(self, setup) -> None:
        features, mask, g = setup
        sub = expand(g, [1, 3], 0, features, mask, iters=3)
        assert 0 in sub

    def anchors(self, setup) -> None:
        features, mask, g = setup
        sub = expand(g, [1, 3], 0, features, mask, iters=5)
        # The query (0) is added; if best-improvement never triggers removal
        # of anchors, they stay. We only assert the query is included.
        assert 0 in sub

    def deterministic(self, setup) -> None:
        features, mask, g = setup
        a = expand(g, [1, 3], 0, features, mask, iters=5)
        b = expand(g, [1, 3], 0, features, mask, iters=5)
        assert a == b

    def iters(self, setup) -> None:
        features, mask, g = setup
        with pytest.raises(Net):
            expand(g, [1, 3], 0, features, mask, iters=0)

    def loop(self, setup) -> None:
        features, mask, _ = setup
        g_loopy = sp.csr_matrix(np.eye(3, dtype=np.float32))
        with pytest.raises(Net, match="graph has self-loops"):
            expand(g_loopy, [0, 2], 0, features, mask[:3], iters=3)
