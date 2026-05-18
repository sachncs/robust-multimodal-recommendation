import numpy as np
import scipy.sparse as sp

from rmr.models.utils import compute_laplacian_pe, normalize_adjacency


def test_normalize_adjacency():
    A = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
    L = normalize_adjacency(A)
    assert L.shape == (3, 3)
    # Symmetric normalized Laplacian should have 1s on diagonal
    assert np.allclose(L.diagonal(), 1.0)


def test_laplacian_pe_shape():
    A = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
    pe = compute_laplacian_pe(A, k=2)
    assert pe.shape == (3, 2)


def test_normalize_adjacency_off_diagonal_symmetry():
    A = sp.csr_matrix(np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
    ], dtype=np.float32))
    L = normalize_adjacency(A)
    dense = L.toarray()
    # Should be symmetric
    np.testing.assert_allclose(dense, dense.T, atol=1e-7)
    # Off-diagonal entries should be non-positive (L = I - D^{-1/2} A D^{-1/2})
    off_diag = dense.copy()
    np.fill_diagonal(off_diag, 0)
    assert np.all(off_diag <= 1e-7)  # allow tiny numerical positives


def test_compute_laplacian_pe_eigenvector_correctness():
    A = sp.csr_matrix(np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=np.float32))
    L = normalize_adjacency(A)
    pe = compute_laplacian_pe(A, k=2)
    # Each eigenvector should approximately satisfy L @ v ≈ λ v
    # We can't easily get λ from the function, but we can verify orthonormality
    # and that L @ v is parallel to v.
    for i in range(pe.shape[1]):
        v = pe[:, i]
        Lv = L.dot(v)
        # Lv should be approximately a scalar multiple of v
        # Compute projection of Lv onto v
        denom = np.dot(v, v)
        if denom > 0:
            proj = np.dot(Lv, v) / denom
            residual = Lv - proj * v
            assert np.linalg.norm(residual) < 1e-4
