"""Tests for morel.graph.laplacian and morel.graph.Laplace."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from morel.core.errors import GraphError
from morel.graph import Laplace, laplacian, pe


def test_laplacian_diagonal_is_one(path3) -> None:
    lap = laplacian(path3)
    diag = lap.diagonal()
    assert np.allclose(diag, 1.0)


def test_laplacian_rejects_self_loops() -> None:
    g = sp.csr_matrix(np.eye(3, dtype=np.float32))
    with pytest.raises(GraphError):
        laplacian(g)


def test_pe_shape_clamps_k() -> None:
    g = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.float32))
    out = pe(g, k=20)
    assert out.shape == (2, 1)


def test_pe_invalid_k() -> None:
    g = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.float32))
    with pytest.raises(GraphError):
        pe(g, k=0)


def test_laplace_caches_same_content(path3) -> None:
    lap = Laplace(k=2)
    a = lap(path3)
    b = lap(path3)
    assert a.data_ptr() == b.data_ptr()


def test_laplace_evicts_when_full(path3) -> None:
    lap = Laplace(k=2, capacity=2)
    lap(path3)
    other = sp.csr_matrix(np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]], dtype=np.float32))
    lap(other)
    assert len(lap.cache) <= 2


def test_laplace_clear(path3) -> None:
    lap = Laplace(k=2)
    lap(path3)
    lap.clear()
    assert len(lap.cache) == 0
