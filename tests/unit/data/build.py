"""Tests for morel.data.build."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from morel.core.errors import DataError
from morel.data.build import bipartite, item_cooccurrence, kcore


class Checker:
    """Aggregated test methods for this module."""

    def bipartite() -> None:
        user = np.array([0, 1, 0, 1])
        item = np.array([0, 1, 2, 3])
        g = bipartite(user, item, 2, 4)
        assert g.shape == (2, 4)
        assert np.allclose(g.toarray(), np.array([[1, 0, 1, 0], [0, 1, 0, 1]]))

    def item() -> None:
        user = np.array([0, 1])
        item = np.array([0, 1])
        g = bipartite(user, item, 2, 2)
        cooc = item_cooccurrence(g)
        diag = cooc.diagonal()
        assert (diag == 0).all()

    def cooccurrence() -> None:
        user = np.array([0, 1, 2, 0, 1])
        item = np.array([0, 1, 2, 1, 2])
        g = bipartite(user, item, 3, 3)
        cooc = item_cooccurrence(g)
        diff = cooc - cooc.T
        assert abs(diff).max() == 0

    def kcore() -> None:
        arr = np.array(
            [
                [0, 1, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0],
            ],
            dtype=np.float32,
        )
        g = sp.csr_matrix(arr)
        sub = kcore(g, min_edges=1)
        assert sub.shape[0] >= 1

    def zero() -> None:
        arr = np.eye(3, dtype=np.float32)
        g = sp.csr_matrix(arr)
        out = kcore(g, min_edges=0)
        assert (out.toarray() == arr).all()

    def rejects() -> None:
        with pytest.raises(DataError):
            bipartite(np.array([0, 1]), np.array([0]), 2, 2)
