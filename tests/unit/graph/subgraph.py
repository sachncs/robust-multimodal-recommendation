"""Tests for morel.graph.subgraph."""

from __future__ import annotations

import numpy as np
import pytest

from morel.core.errors import Net
from morel.graph import Subgraph


class Checker:
    """Aggregated test methods for this module."""

    def sort(self) -> None:
        sg = Subgraph.from_idx([3, 1, 2, 1, 3])
        assert sg.nodes.tolist() == [1, 2, 3]

    def contains(self) -> None:
        sg = Subgraph.from_idx([0, 1])
        assert 1 in sg
        assert 99 not in sg

    def rejected(self) -> None:
        with pytest.raises(Net):
            Subgraph(nodes=np.array([-1, 0]))

    def iter(self) -> None:
        sg = Subgraph.from_idx([0, 1, 2])
        assert list(sg) == [0, 1, 2]