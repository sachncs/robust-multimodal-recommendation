"""Tests for morel.retrieve.bfs."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from morel.core.errors import GraphError
from morel.retrieve.bfs import bfs, neighbor_array, neighbor_iter, path


def test_bfs_distances_on_path() -> None:
    g = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
    d = bfs(g, [0])
    assert d == {0: 0, 1: 1, 2: 2}


def test_bfs_multi_source() -> None:
    g = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
    d = bfs(g, [0, 2])
    assert d[1] == 1


def test_path_finds_shortest() -> None:
    g = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
    assert path(g, 0, 2) == [0, 1, 2]


def test_path_unreachable() -> None:
    g = sp.csr_matrix(np.array([[0, 0], [0, 0]], dtype=np.float32))
    assert path(g, 0, 1) == []


def test_path_self() -> None:
    g = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.float32))
    assert path(g, 0, 0) == [0]


def test_path_out_of_range() -> None:
    g = sp.csr_matrix(np.eye(3, dtype=np.float32))
    with pytest.raises(GraphError):
        path(g, 0, 99)


def test_neighbor_array() -> None:
    g = sp.csr_matrix(np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.float32))
    out = neighbor_array(g)
    assert out[0].tolist() == [1, 2]
    assert out[1].tolist() == [0]


def test_neighbor_iter() -> None:
    g = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32))
    assert sorted(neighbor_iter(g, 0)) == [1]
