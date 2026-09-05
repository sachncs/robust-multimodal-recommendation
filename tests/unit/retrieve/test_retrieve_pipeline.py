"""Tests for morel.retrieve.pipeline."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from morel.retrieve.pipeline import batch, retrieve


def _path(n: int) -> sp.csr_matrix:
    arr = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        arr[i, i + 1] = 1
        arr[i + 1, i] = 1
    return sp.csr_matrix(arr)


def test_retrieve_includes_query() -> None:
    g = _path(5)
    features = {"visual": np.eye(5, dtype=np.float32), "text": np.eye(5, dtype=np.float32)}
    mask = np.ones((5, 2), dtype=np.float32)
    sub = retrieve(0, features, mask, g, anchors=2, iters=2)
    assert 0 in sub


def test_batch_returns_result() -> None:
    g = _path(5)
    features = {"visual": np.eye(5, dtype=np.float32)}
    mask = np.ones((5, 1), dtype=np.float32)
    result = batch([0, 1], features, mask, g, anchors=2, iters=2)
    assert result.batch == 2
    assert (result.sizes > 0).all()


def test_batch_deterministic() -> None:
    g = _path(5)
    features = {"visual": np.eye(5, dtype=np.float32), "text": np.eye(5, dtype=np.float32)}
    mask = np.ones((5, 2), dtype=np.float32)
    a = batch([0, 1], features, mask, g, anchors=2, iters=3)
    b = batch([0, 1], features, mask, g, anchors=2, iters=3)
    assert (a.nodes == b.nodes).all()
    assert (a.mask == b.mask).all()


def test_empty_modalities_returns_query_only() -> None:
    g = _path(3)
    features = {"visual": np.eye(3, dtype=np.float32), "text": np.eye(3, dtype=np.float32)}
    mask = np.zeros((3, 2), dtype=np.float32)
    sub = retrieve(0, features, mask, g, anchors=1, iters=1)
    assert sub == {0}
