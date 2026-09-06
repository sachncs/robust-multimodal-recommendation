"""Tests for morel.graph.bipartite and morel.graph.item."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from morel.graph import Bipartite, Item, connected


class Checker:
    """Aggregated test methods for this module."""

    def constructs(self) -> None:
        g = Bipartite(sp.csr_matrix(np.eye(2, dtype=np.float32)))
        assert g.users == 2
        assert g.items == 2
        assert g.edges == 2

    def loops(self) -> None:
        bi = Bipartite(sp.csr_matrix(np.array([[1, 1], [1, 1]], dtype=np.float32)))
        item = Item.from_bip(bi)
        assert (item.matrix.diagonal() == 0).all()

    def symmetric(self) -> None:
        bi = Bipartite(sp.csr_matrix(np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.float32)))
        item = Item.from_bip(bi)
        assert (item.matrix - item.matrix.T).max() == 0

    def self(self) -> None:
        from morel.core.errors import Net

        g = sp.csr_matrix(np.eye(3, dtype=np.float32))
        with pytest.raises(Net):
            Item(matrix=g)

    def connected(self) -> None:
        nodes = np.array([0, 1, 2])
        parent = {0: np.array([1]), 1: np.array([0, 2]), 2: np.array([1])}
        assert connected(nodes, parent)

    def disconnected(self) -> None:
        nodes = np.array([0, 2])
        parent = {0: np.array([1]), 1: np.array([0, 2]), 2: np.array([1])}
        assert not connected(nodes, parent)
