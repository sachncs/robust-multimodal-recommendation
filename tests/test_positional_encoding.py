import numpy as np
import scipy.sparse as sp
import torch

from rmr.models.positional_encoding import LaplacianPE


def test_laplacian_pe_shape():
    adj = sp.csr_matrix(np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=np.float32))
    pe = LaplacianPE(k=2)
    emb = pe(adj)
    assert emb.shape == (3, 2)


def test_caching_idempotency():
    adj = sp.csr_matrix(np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=np.float32))
    pe = LaplacianPE(k=2)
    emb1 = pe(adj)
    emb2 = pe(adj)
    # Should return the exact same cached tensor object
    assert emb1 is emb2


def test_clear_cache():
    adj = sp.csr_matrix(np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=np.float32))
    pe = LaplacianPE(k=2)
    emb1 = pe(adj)
    pe.clear_cache()
    assert len(pe._cache) == 0
    emb2 = pe(adj)
    # After clearing cache, we get a new tensor (values same, object different)
    assert emb1 is not emb2
    torch.testing.assert_close(emb1, emb2)


def test_eigsh_path():
    # Build a graph with n=10, k=2, so n > k+1 triggers eigsh path
    n = 10
    row = list(range(n - 1)) + list(range(1, n))
    col = list(range(1, n)) + list(range(n - 1))
    data = np.ones(len(row), dtype=np.float32)
    adj = sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    pe = LaplacianPE(k=2)
    emb = pe(adj)
    assert emb.shape == (n, 2)
    assert torch.isfinite(emb).all()
