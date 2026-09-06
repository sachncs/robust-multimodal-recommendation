"""Tests for morel.data.validate."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from morel.core.errors import Datum
from morel.data.validate import features, graph, interactions
from morel.data.validate import mask as check


class Checker:
    """Aggregated test methods for this module."""

    def ok(self) -> None:
        interactions(np.array([0, 1]), np.array([0, 1]), 2, 2)

    def user(self) -> None:
        with pytest.raises(Datum):
            interactions(np.array([5]), np.array([0]), 2, 2)

    def nan(self) -> None:
        arr = np.array([[1.0, 2.0], [np.nan, 0.0]], dtype=np.float32)
        with pytest.raises(Datum):
            features({"a": arr}, items=2)

    def dtype(self) -> None:
        arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
        with pytest.raises(Datum):
            features({"a": arr}, items=2)

    def loops(self) -> None:
        g = sp.csr_matrix(np.array([[1, 1], [1, 0]], dtype=np.float32))
        with pytest.raises(Datum):
            graph(g)

    def negative(self) -> None:
        g = sp.csr_matrix(np.array([[0, -1], [-1, 0]], dtype=np.float32))
        with pytest.raises(Datum):
            graph(g)

    def kept(self) -> None:
        with pytest.raises(Datum):
            check(np.array([[0, 0], [1, 1]], dtype=np.float32))

    def binary(self) -> None:
        with pytest.raises(Datum):
            check(np.array([[0.5, 1.0]], dtype=np.float32))