"""Tests for morel.retrieve.bfs."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from morel.core.errors import GraphError
from morel.retrieve.bfs import bfs, neighbor_array, neighbor_iter, path


class Checker:
    """Aggregated test methods for this module."""

    def path(self) -> None:
        g = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
        d = bfs(g, [0])
        assert d == {0: 0, 1: 1, 2: 2}

    def source(self) -> None:
        g = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
        d = bfs(g, [0, 2])
        assert d[1] == 1

    def shortest(self) -> None:
        g = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
        assert path(g, 0, 2) == [0, 1, 2]

    def unreachable(self) -> None:
        g = sp.csr_matrix(np.array([[0, 0], [0, 0]], dtype=np.float32))
        assert path(g, 0, 1) == []

    def self(self) -> None:
        g = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.float32))
        assert path(g, 0, 0) == [0]

    def range(self) -> None:
        g = sp.csr_matrix(np.eye(3, dtype=np.float32))
        with pytest.raises(GraphError):
            path(g, 0, 99)

    def array(self) -> None:
        g = sp.csr_matrix(np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.float32))
        out = neighbor_array(g)
        assert out[0].tolist() == [1, 2]
        assert out[1].tolist() == [0]

    def iter(self) -> None:
        g = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32))
        assert sorted(neighbor_iter(g, 0)) == [1]