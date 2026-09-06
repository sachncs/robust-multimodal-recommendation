"""Tests for morel.graph.bipartite and morel.graph.item."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from morel.graph import Bipartite, Item, connected


def test_bipartite_constructs() -> None:
    g = Bipartite(sp.csr_matrix(np.eye(2, dtype=np.float32)))
    assert g.users == 2
    assert g.items == 2
    assert g.edges == 2


def test_item_no_self_loops() -> None:
    bi = Bipartite(sp.csr_matrix(np.array([[1, 1], [1, 1]], dtype=np.float32)))
    item = Item.from_bipartite(bi)
    assert (item.matrix.diagonal() == 0).all()


def test_item_is_symmetric() -> None:
    bi = Bipartite(sp.csr_matrix(np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.float32)))
    item = Item.from_bipartite(bi)
    assert (item.matrix - item.matrix.T).max() == 0


def test_item_rejects_self_loops() -> None:
    from morel.core.errors import GraphError

    g = sp.csr_matrix(np.eye(3, dtype=np.float32))
    with pytest.raises(GraphError):
        Item(matrix=g)


def test_connected() -> None:
    nodes = np.array([0, 1, 2])
    parent = {0: np.array([1]), 1: np.array([0, 2]), 2: np.array([1])}
    assert connected(nodes, parent)


def test_disconnected() -> None:
    nodes = np.array([0, 2])
    parent = {0: np.array([1]), 1: np.array([0, 2]), 2: np.array([1])}
    assert not connected(nodes, parent)
