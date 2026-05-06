import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh


def normalize_adjacency(adj: sp.csr_matrix) -> sp.csr_matrix:
    """Compute symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return sp.eye(adj.shape[0]) - D_inv_sqrt @ adj @ D_inv_sqrt


def compute_laplacian_pe(adj: sp.csr_matrix, k: int = 20) -> np.ndarray:
    """Compute bottom-k nontrivial eigenvectors of normalized Laplacian."""
    L = normalize_adjacency(adj)
    n = L.shape[0]
    # Compute k+1 smallest eigenvalues, drop the trivial (smallest) one
    if k + 1 >= n:
        eigvals, eigvecs = eigh(L.toarray())
    else:
        eigvals, eigvecs = eigsh(L, k=k + 1, which="SM")
    return eigvecs[:, 1 : k + 1]
