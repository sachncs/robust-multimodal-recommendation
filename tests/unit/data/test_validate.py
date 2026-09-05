"""Tests for morel.data.validate."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from morel.core.errors import DataError
from morel.data.validate import features, graph, interactions, mask as validate_mask


def test_interactions_ok() -> None:
    interactions(np.array([0, 1]), np.array([0, 1]), 2, 2)


def test_interactions_out_of_range_user() -> None:
    with pytest.raises(DataError):
        interactions(np.array([5]), np.array([0]), 2, 2)


def test_features_rejects_nan() -> None:
    arr = np.array([[1.0, 2.0], [np.nan, 0.0]], dtype=np.float32)
    with pytest.raises(DataError):
        features({"a": arr}, items=2)


def test_features_rejects_wrong_dtype() -> None:
    arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
    with pytest.raises(DataError):
        features({"a": arr}, items=2)


def test_graph_rejects_self_loops() -> None:
    g = sp.csr_matrix(np.array([[1, 1], [1, 0]], dtype=np.float32))
    with pytest.raises(DataError):
        graph(g)


def test_graph_rejects_negative() -> None:
    g = sp.csr_matrix(np.array([[0, -1], [-1, 0]], dtype=np.float32))
    with pytest.raises(DataError):
        graph(g)


def test_mask_rejects_no_kept() -> None:
    with pytest.raises(DataError):
        validate_mask(np.array([[0, 0], [1, 1]], dtype=np.float32))


def test_mask_rejects_non_binary() -> None:
    with pytest.raises(DataError):
        validate_mask(np.array([[0.5, 1.0]], dtype=np.float32))
