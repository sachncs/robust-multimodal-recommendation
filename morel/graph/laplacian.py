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

from morel.core.errors import Net
from morel.core.log import get as get_logger
from morel.graph import invariants

log = get_logger("graph.laplacian")

#: Seed for the fixed ARPACK start vector. Any constant works; it only has to
#: be the same on every run so the Lanczos iteration is reproducible.
SEED = 0


def start(nodes: int) -> np.ndarray:
    """Return a fixed ARPACK start vector for a graph with ``nodes`` nodes.

    ``scipy.sparse.linalg.eigsh`` draws ``v0`` from numpy's global RNG when the
    caller does not supply one, which makes the resulting eigenvectors depend
    on ambient process state. Supplying this vector removes that dependence.
    """
    return np.random.default_rng(SEED).standard_normal(nodes)


def coo_hash(matrix: sp.spmatrix) -> str:
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
    invariants.no_loops(adj)
    adj_coo = sp.coo_matrix(adj)
    rowsum = np.asarray(adj_coo.sum(axis=1)).flatten()
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(rowsum > 0, np.power(rowsum, -0.5), 0.0)
    d_inv_sqrt = sp.diags(d_inv_sqrt.astype(np.float32))
    laplacian_csr = sp.eye(adj.shape[0], format="csr") - d_inv_sqrt @ adj_coo @ d_inv_sqrt
    return laplacian_csr.tocsr()


def signs(eigvecs: np.ndarray) -> np.ndarray:
    """Fix the arbitrary sign of each eigenvector to a reproducible choice.

    An eigenvector is only defined up to sign, so two mathematically correct
    solvers (or two runs of the same iterative solver) may return columns that
    differ by a factor of -1. Downstream positional encodings would then differ
    for identical input. This orients each column so that its
    largest-magnitude entry is positive, which is stable under both the dense
    and the ARPACK code paths.

    Args:
        eigvecs: ``(nodes, k)`` array of column eigenvectors.

    Returns
    -------
        The same array with each column sign-normalized.
    """
    if eigvecs.size == 0:
        return eigvecs
    pivot = np.abs(eigvecs).argmax(axis=0)
    signs = np.sign(eigvecs[pivot, np.arange(eigvecs.shape[1])])
    signs[signs == 0] = 1.0
    return eigvecs * signs


#: Eigenvalue gap below which two modes are treated as one degenerate cluster.
#: This has to be comfortably larger than the error in ARPACK's computed
#: eigenvalues (which drifts between runs, around 1e-8 here) and comfortably
#: smaller than the real gaps between distinct eigenvalues (around 5e-3 for the
#: graphs this operates on). If the tolerance is too tight, a genuine
#: degenerate pair intermittently splits into two singleton clusters, which
#: skips canonicalization and lets the basis rotate freely between runs.
TOL = 1e-6


def basis(eigvals: np.ndarray, eigvecs: np.ndarray, *, tol: float = TOL) -> np.ndarray:
    """Pick a reproducible basis inside each degenerate eigenspace.

    Sign-fixing is not enough when an eigenvalue is repeated: any orthogonal
    rotation of the columns spanning that eigenspace is an equally valid
    answer, so two solver runs can return genuinely different (not merely
    sign-flipped) vectors. Graphs hit this constantly — every isolated node
    contributes to the same eigenvalue, and any symmetry does too.

    Within each group of near-equal eigenvalues this projects a fixed
    reference frame onto the eigenspace and re-orthonormalizes. The output
    then depends only on the subspace itself, which the solver determines
    uniquely, and not on the arbitrary basis the solver happened to return.

    Args:
        eigvals: ``(k,)`` eigenvalues in ascending order.
        eigvecs: ``(nodes, k)`` matching column eigenvectors.
        tol: Absolute eigenvalue gap below which two consecutive modes are
            treated as belonging to the same degenerate cluster.

    Returns
    -------
        ``(nodes, k)`` eigenvectors with a canonical basis per eigenspace.
    """
    nodes, k = eigvecs.shape
    if k == 0:
        return eigvecs
    out = eigvecs.copy()
    reference = np.random.default_rng(SEED).standard_normal((nodes, k))
    start = 0
    while start < k:
        stop = start + 1
        while stop < k and abs(float(eigvals[stop]) - float(eigvals[start])) <= tol:
            stop += 1
        if stop - start > 1:
            block = eigvecs[:, start:stop]
            # Re-express a fixed reference frame inside this eigenspace.
            projected = block @ (block.T @ reference[:, start:stop])
            q, r = np.linalg.qr(projected)
            diagonal = np.diag(r).copy()
            diagonal[diagonal == 0] = 1.0
            out[:, start:stop] = q * np.sign(diagonal)
        start = stop
    return out


#: Node count up to which the dense, direct eigensolver is used.
#:
#: ARPACK is iterative: on a degenerate spectrum its eigenvectors are not
#: reproducible between runs even with a pinned start vector, because tiny
#: run-to-run drift in the computed eigenvalues rotates the basis within a
#: repeated eigenspace. ``numpy.linalg.eigh`` is a direct method and is
#: bitwise reproducible for identical input. The positional encoding is
#: computed once per graph and cached, so paying O(n^3) once on a graph of
#: this size is a good trade for a reproducible result. Above the threshold
#: the dense path would be too costly and the sparse solver is used instead;
#: see :func:`pe` for what that means for reproducibility.
LIMIT = 2048


def straddles(eigvals: np.ndarray, want: int, *, tol: float = TOL) -> bool:
    """Return whether a degenerate cluster is cut in half by the ``want`` cutoff.

    If the last kept eigenvalue equals the first discarded one, the solver has
    handed back an arbitrary subset of a larger eigenspace and the result
    cannot be made reproducible without seeing the rest of that cluster.
    """
    if want >= eigvals.shape[0]:
        return False
    return bool(abs(float(eigvals[want]) - float(eigvals[want - 1])) <= tol)


def bottom(lap: sp.csr_matrix, count: int, nodes: int, *, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``count`` smallest eigenpairs, sparsely when that is possible.

    Small graphs go through the dense solver because it is deterministic;
    ARPACK cannot return every eigenpair of a matrix anyway, so requests that
    approach the matrix size also use the dense path.
    """
    if count >= nodes or nodes <= LIMIT:
        dense_vals, dense_vecs = np.linalg.eigh(lap.toarray())
        return dense_vals, dense_vecs
    try:
        sparse_vals, sparse_vecs = spla.eigsh(lap, k=count, which="SM", v0=start(nodes))
    except spla.ArpackNoConvergence as exc:
        log.warning(
            "eigsh did not converge; falling back to dense eigh",
            extra={"k": k, "nodes": nodes, "reason": str(exc)},
        )
        dense_vals, dense_vecs = np.linalg.eigh(lap.toarray())
        return dense_vals, dense_vecs
    return sparse_vals, sparse_vecs


def pe(adj: sp.spmatrix, k: int = 20) -> np.ndarray:
    """Compute the bottom-k nontrivial Laplacian PE.

    Uses Lanczos/ARPACK for sparse matrices and dense ``eigh`` for small ones.
    Catches only ``ArpackNoConvergence``; all other exceptions propagate.

    The result is deterministic. Four things are pinned that otherwise vary
    between runs: ARPACK's start vector (it would be drawn from numpy's global
    RNG), the ordering of the returned eigenpairs, the basis chosen inside
    degenerate eigenspaces, and the arbitrary sign of each eigenvector.

    A degenerate eigenspace only has a canonical basis if the *whole* cluster
    is available. If the requested cutoff falls inside a repeated eigenvalue —
    common when a graph has many isolated nodes, which all share one
    eigenvalue — the solver would otherwise return an arbitrary slice of a
    much larger eigenspace, and no post-processing can make that reproducible.
    This widens the request until the boundary cluster is whole.

    Graphs of up to :data:`LIMIT` nodes use the direct dense solver,
    which is bitwise reproducible. Larger graphs use ARPACK, where the
    canonicalization above removes the arbitrary signs and the rotation within
    fully-captured degenerate clusters, but the iterative solver's own drift
    means agreement is to floating-point precision rather than bitwise.

    Args:
        adj: Symmetric item-item CSR adjacency.
        k: Number of nontrivial eigenvectors.

    Returns
    -------
        Array of shape ``(nodes, min(k, n-1))`` with the bottom eigenvectors
        excluding the trivial constant one. ``k`` is clamped to ``n - 1``.
    """
    if k < 0:
        raise Net(f"k must be non-negative, got {k}")
    if k == 0:
        # Zero requested dimensions means no positional encoding at all. This
        # is a meaningful configuration, not an error: it is the "no_pe"
        # ablation condition. Callers concatenate the result, and a
        # zero-width array contributes nothing.
        return np.zeros((adj.shape[0], 0), dtype=np.float64)
    lap = laplacian(adj)
    nodes = lap.shape[0]
    if nodes <= 1:
        return np.zeros((nodes, k), dtype=np.float64)
    effective_k = min(k, nodes - 1)
    if effective_k <= 0:
        return np.zeros((nodes, 0), dtype=np.float64)
    want = effective_k + 1

    # Request one extra eigenpair beyond the cutoff so that a degenerate
    # cluster sitting on the boundary is visible at all; without the probe
    # there is no "next" eigenvalue to compare the last kept one against.
    requested = min(nodes, want + 1)
    while True:
        eigvals, eigvecs = bottom(lap, requested, nodes, k=k)
        order = np.argsort(eigvals)
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        if eigvals.shape[0] >= nodes or not straddles(eigvals, want):
            break
        widened = min(nodes, requested * 2)
        if widened == requested:
            break
        requested = widened

    eigvecs = basis(eigvals, eigvecs)
    return signs(eigvecs[:, 1:want]).astype(np.float64)


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
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def forward(self, adjacency: sp.spmatrix) -> torch.Tensor:
        """Compute or retrieve cached Laplacian PE."""
        key = coo_hash(adjacency)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        array = pe(adjacency, k=self.k)
        tensor = torch.from_numpy(array).float()
        self.cache[key] = tensor
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return tensor

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()


__all__ = [
    "LIMIT",
    "TOL",
    "Laplace",
    "basis",
    "bottom",
    "laplacian",
    "pe",
    "signs",
    "start",
    "straddles",
]
