"""Tests for morel.retrieve.acs."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from morel.core.errors import Net
from morel.retrieve.acs import batch, compute


class Checker:
    """Aggregated test methods for this module."""

    def path(self) -> None:
        g = sp.csr_matrix(
            np.array(
                [[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0]],
                dtype=np.float32,
            )
        )
        sub = compute(g, [0, 4])
        assert sub == {0, 1, 2, 3, 4}

    def anchor(self) -> None:
        g = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
        assert compute(g, [0]) == {0}

    def empty(self) -> None:
        g = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
        assert compute(g, []) == set()

    def raises(self) -> None:
        g = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
        with pytest.raises(Net):
            compute(g, [0, 0])

    def range(self) -> None:
        g = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
        with pytest.raises(Net):
            compute(g, [0, 99])

    def loops(self) -> None:
        g = sp.csr_matrix(np.eye(3, dtype=np.float32))
        with pytest.raises(Net):
            compute(g, [0, 2])

    def fallback(self) -> None:
        g = sp.csr_matrix(
            np.array(
                [[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]],
                dtype=np.float32,
            )
        )
        sub = compute(g, [0, 3], fallback="empty")
        assert sub == set()

    def paths(self) -> None:
        n = 4000
        rows = []
        for i in range(n - 1):
            rows.append((i, i + 1))
        data = np.ones(2 * (n - 1), dtype=np.float32)
        row_arr = np.array([r for r, _ in rows] + [c for _, c in rows], dtype=np.int32)
        col_arr = np.array([c for r, c in rows] + [r for r, _ in rows], dtype=np.int32)
        g = sp.csr_matrix((data, (row_arr, col_arr)), shape=(n, n))
        sub = compute(g, [0, n - 1])
        assert 0 in sub
        assert (n - 1) in sub
        assert len(sub) == n

    def batch(self) -> None:
        g = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
        out = batch(g, [[0], [1]])
        assert out == [{0}, {1}]