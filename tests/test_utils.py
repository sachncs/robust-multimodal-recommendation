import numpy as np
import scipy.sparse as sp
from models.utils import normalize_adjacency, compute_laplacian_pe


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
