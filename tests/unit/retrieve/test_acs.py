"""Tests for morel.retrieve.acs."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from morel.core.errors import GraphError
from morel.retrieve.acs import batch, compute


def test_acs_two_anchors_on_path() -> None:
    g = sp.csr_matrix(
        np.array(
            [[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0]],
            dtype=np.float32,
        )
    )
    sub = compute(g, [0, 4])
    assert sub == {0, 1, 2, 3, 4}


def test_acs_single_anchor() -> None:
    g = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
    assert compute(g, [0]) == {0}


def test_acs_no_anchors_returns_empty() -> None:
    g = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
    assert compute(g, []) == set()


def test_acs_duplicate_anchor_raises() -> None:
    g = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
    with pytest.raises(GraphError):
        compute(g, [0, 0])


def test_acs_out_of_range_raises() -> None:
    g = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
    with pytest.raises(GraphError):
        compute(g, [0, 99])


def test_acs_self_loops_raises() -> None:
    g = sp.csr_matrix(np.eye(3, dtype=np.float32))
    with pytest.raises(GraphError):
        compute(g, [0, 2])


def test_acs_fallback_empty() -> None:
    g = sp.csr_matrix(
        np.array(
            [[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]],
            dtype=np.float32,
        )
    )
    sub = compute(g, [0, 3], fallback="empty")
    assert sub == set()


def test_acs_iterative_handles_long_paths() -> None:
    n = 4000
    rows = []
    for i in range(n - 1):
        rows.append((i, i + 1))
    data = np.ones(2 * (n - 1), dtype=np.float32)
    row_arr = np.array([r for r, _ in rows] + [c for _, c in rows], dtype=np.int32)
    col_arr = np.array([c for r, c in rows] + [r for r, _ in rows], dtype=np.int32)
    g = sp.csr_matrix((data, (row_arr, col_arr)), shape=(n, n))
    sub = compute(g, [0, n - 1])
    assert 0 in sub and (n - 1) in sub and len(sub) == n


def test_acs_batch() -> None:
    g = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
    out = batch(g, [[0], [1]])
    assert out == [{0}, {1}]
