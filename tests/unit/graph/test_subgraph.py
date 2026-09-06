"""Tests for morel.graph.subgraph."""

from __future__ import annotations

import numpy as np
import pytest

from morel.core.errors import GraphError
from morel.graph import Subgraph


class Checker:
    """Aggregated test methods for this module."""

    def subgraph() -> None:
        sg = Subgraph.from_indices([3, 1, 2, 1, 3])
        assert sg.nodes.tolist() == [1, 2, 3]

    def contains() -> None:
        sg = Subgraph.from_indices([0, 1])
        assert 1 in sg
        assert 99 not in sg

    def negative() -> None:
        with pytest.raises(GraphError):
            Subgraph(nodes=np.array([-1, 0]))

    def verify() -> None:
        sg = Subgraph.from_indices([0, 1, 2])
        assert list(sg) == [0, 1, 2]
