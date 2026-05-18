"""Graph utility functions for normalized Laplacian and positional encodings."""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh


def normalize_adjacency(adj: sp.csr_matrix) -> sp.csr_matrix:
    """Compute the symmetric normalized Laplacian.

    Calculates L = I - D^{-1/2} A D^{-1/2}, where A is the adjacency
    matrix and D is the degree matrix.

    Args:
        adj: Sparse adjacency matrix in CSR format.

    Returns:
        The symmetric normalized Laplacian matrix in CSR format.
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.zeros_like(rowsum)
    d_inv_sqrt[rowsum > 0] = np.power(rowsum[rowsum > 0], -0.5)
    d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
    return sp.eye(adj.shape[0]) - d_inv_sqrt_mat @ adj @ d_inv_sqrt_mat


def compute_laplacian_pe(
    adj: sp.csr_matrix, k: int = 20
) -> np.ndarray:
    """Compute the bottom-k nontrivial eigenvectors of the normalized Laplacian.

    Args:
        adj: Sparse adjacency matrix in CSR format.
        k: Number of eigenvectors to compute (excluding the trivial one).

    Returns:
        Array of shape (n_nodes, k) containing the bottom-k nontrivial
        eigenvectors of the normalized Laplacian.
    """
    laplacian = normalize_adjacency(adj)
    n = laplacian.shape[0]
    if k + 1 >= n:
        eigvals, eigvecs = eigh(laplacian.toarray())
    else:
        try:
            eigvals, eigvecs = eigsh(laplacian, k=k + 1, which="SM")
        except Exception:
            eigvals, eigvecs = eigh(laplacian.toarray())
    return eigvecs[:, 1 : k + 1]
