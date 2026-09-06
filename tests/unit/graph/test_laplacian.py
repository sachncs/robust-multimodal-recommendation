"""Tests for morel.graph.laplacian and morel.graph.Laplace."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from morel.core.errors import GraphError
from morel.graph.laplacian import Laplace, canonical_signs, laplacian, pe, start_vector


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


def test_pe_negative_k_is_rejected() -> None:
    g = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.float32))
    with pytest.raises(GraphError, match="k must be non-negative"):
        pe(g, k=-1)


def test_pe_zero_k_means_no_positional_encoding() -> None:
    """k=0 is the "no_pe" ablation condition, not an error."""
    g = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.float32))
    out = pe(g, k=0)
    assert out.shape == (2, 0)


def test_laplace_caches_same_content(path3) -> None:
    lap = Laplace(k=2)
    a = lap(path3)
    b = lap(path3)
    assert a.data_ptr() == b.data_ptr()


def test_laplace_evicts_when_full(path3) -> None:
    lap = Laplace(k=2, capacity=2)
    lap(path3)
    other = sp.csr_matrix(
        np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]], dtype=np.float32)
    )
    lap(other)
    assert len(lap.cache) <= 2


def test_laplace_clear(path3) -> None:
    lap = Laplace(k=2)
    lap(path3)
    lap.clear()
    assert len(lap.cache) == 0


def ring_graph(nodes: int) -> sp.csr_matrix:
    """Undirected ring, large enough to exercise the sparse ARPACK path."""
    arr = np.zeros((nodes, nodes), dtype=np.float32)
    for i in range(nodes):
        arr[i, (i + 1) % nodes] = 1.0
        arr[(i + 1) % nodes, i] = 1.0
    return sp.csr_matrix(arr)


def test_pe_is_independent_of_global_numpy_rng() -> None:
    """Regression: eigsh drew its start vector from the global RNG."""
    adj = ring_graph(60)
    np.random.seed(1)
    first = pe(adj, k=20)
    np.random.seed(9999)
    second = pe(adj, k=20)
    assert np.array_equal(first, second)


def test_pe_repeated_calls_agree() -> None:
    adj = ring_graph(60)
    assert np.array_equal(pe(adj, k=20), pe(adj, k=20))


def test_canonical_signs_orients_largest_entry_positive() -> None:
    vecs = np.array([[0.1, -0.9], [-0.8, 0.2], [0.3, 0.4]])
    out = canonical_signs(vecs)
    pivot = np.abs(out).argmax(axis=0)
    assert np.all(out[pivot, np.arange(out.shape[1])] > 0)


def test_canonical_signs_is_idempotent() -> None:
    vecs = np.array([[0.1, -0.9], [-0.8, 0.2], [0.3, 0.4]])
    once = canonical_signs(vecs)
    assert np.array_equal(once, canonical_signs(once))


def test_canonical_signs_handles_empty() -> None:
    empty = np.zeros((5, 0))
    assert canonical_signs(empty).shape == (5, 0)


def test_start_vector_is_stable() -> None:
    assert np.array_equal(start_vector(16), start_vector(16))


def test_ring_graph_eigenvalues_are_degenerate() -> None:
    """Guard the premise of the degeneracy tests below."""
    lap = laplacian(ring_graph(60)).toarray()
    eigvals = np.linalg.eigvalsh(lap)
    assert np.sum(np.abs(np.diff(eigvals)) < 1e-9) > 0


def test_pe_is_stable_under_eigenvalue_degeneracy() -> None:
    """Sign-fixing alone cannot pin a basis inside a repeated eigenspace."""
    adj = ring_graph(60)
    runs = [pe(adj, k=20) for _ in range(4)]
    for other in runs[1:]:
        assert np.array_equal(runs[0], other)


def test_pe_is_stable_with_many_isolated_nodes() -> None:
    """Isolated nodes create a large degenerate eigenspace; common in item graphs."""
    arr = np.zeros((80, 80), dtype=np.float32)
    for i in range(10):
        arr[i, i + 1] = arr[i + 1, i] = 1.0
    adj = sp.csr_matrix(arr)
    runs = [pe(adj, k=20) for _ in range(4)]
    for other in runs[1:]:
        assert np.array_equal(runs[0], other)


def test_pe_returns_genuine_eigenvectors() -> None:
    """Canonicalization must not disturb the eigenvector property."""
    adj = ring_graph(60)
    lap = laplacian(adj).toarray()
    vectors = pe(adj, k=20)
    for column in range(vectors.shape[1]):
        vector = vectors[:, column]
        rayleigh = vector @ lap @ vector / (vector @ vector)
        assert np.linalg.norm(lap @ vector - rayleigh * vector) < 1e-8


def test_pe_columns_are_orthonormal() -> None:
    vectors = pe(ring_graph(60), k=20)
    gram = vectors.T @ vectors
    assert np.abs(gram - np.eye(vectors.shape[1])).max() < 1e-8
