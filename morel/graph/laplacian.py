"""Symmetric normalized Laplacian and bottom-k Laplacian PE.

The cache key for ``Laplace`` is a content-derived hash so that re-binding to
a different adjacency object does not silently return a stale PE.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn

from morel.core.errors import GraphError
from morel.core.log import get as get_logger
from morel.graph import invariants

log = get_logger("graph.laplacian")


def _coo_data_hash(matrix: sp.spmatrix) -> str:
    """Stable hash of a sparse matrix's nonzero pattern and shape."""
    coo = sp.coo_matrix(matrix)
    h = hashlib.sha256()
    h.update(np.asarray(coo.shape, dtype=np.int64).tobytes())
    h.update(coo.data.astype(np.float32, copy=False).tobytes())
    h.update(coo.row.astype(np.int64, copy=False).tobytes())
    h.update(coo.col.astype(np.int64, copy=False).tobytes())
    return h.hexdigest()


def laplacian(adj: sp.spmatrix) -> sp.csr_matrix:
    """Symmetric normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}."""
    invariants.no_self_loops(adj)
    adj_coo = sp.coo_matrix(adj)
    rowsum = np.asarray(adj_coo.sum(axis=1)).flatten()
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(rowsum > 0, np.power(rowsum, -0.5), 0.0)
    d_inv_sqrt = sp.diags(d_inv_sqrt.astype(np.float32))
    laplacian_csr = sp.eye(adj.shape[0], format="csr") - d_inv_sqrt @ adj_coo @ d_inv_sqrt
    return laplacian_csr.tocsr()


def pe(adj: sp.spmatrix, k: int = 20) -> np.ndarray:
    """Compute the bottom-k nontrivial Laplacian PE.

    Uses Lanczos/ARPACK for sparse matrices and dense ``eigh`` for small ones.
    Catches only ``ArpackNoConvergence``; all other exceptions propagate.

    Args:
        adj: Symmetric item-item CSR adjacency.
        k: Number of nontrivial eigenvectors.

    Returns
    -------
        Array of shape ``(nodes, min(k, n-1))`` with the bottom eigenvectors
        excluding the trivial constant one. ``k`` is clamped to ``n - 1``.
    """
    if k <= 0:
        raise GraphError(f"k must be positive, got {k}")
    lap = laplacian(adj)
    nodes = lap.shape[0]
    if nodes <= 1:
        return np.zeros((nodes, k), dtype=np.float64)
    effective_k = min(k, nodes - 1)
    if effective_k <= 0:
        return np.zeros((nodes, 0), dtype=np.float64)
    if effective_k + 1 >= nodes:
        dense = lap.toarray()
        _, eigvecs = np.linalg.eigh(dense)
        return eigvecs[:, 1 : effective_k + 1].astype(np.float64)
    try:
        _, eigvecs = spla.eigsh(lap, k=effective_k + 1, which="SM")
    except spla.ArpackNoConvergence as exc:
        log.warning(
            "eigsh did not converge; falling back to dense eigh",
            extra={"k": k, "nodes": nodes, "reason": str(exc)},
        )
        _, eigvecs = np.linalg.eigh(lap.toarray())
    return eigvecs[:, 1 : effective_k + 1].astype(np.float64)


class Laplace(nn.Module):
    """Caching bottom-k Laplacian PE module.

    Cache key is a content-derived SHA256 so that re-binding a different
    adjacency cannot return a stale tensor (the original bug used ``id()``
    which recycles with GC).
    """

    def __init__(self, k: int = 20, *, capacity: int = 8) -> None:
        super().__init__()
        self.k = k
        self.capacity = capacity
        self._cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    def forward(self, adjacency: sp.spmatrix) -> torch.Tensor:
        """Compute or retrieve cached Laplacian PE."""
        key = _coo_data_hash(adjacency)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        array = pe(adjacency, k=self.k)
        tensor = torch.from_numpy(array).float()
        self._cache[key] = tensor
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return tensor

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()


__all__ = ["Laplace", "laplacian", "pe"]
