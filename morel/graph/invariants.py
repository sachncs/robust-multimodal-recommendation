"""Graph invariants asserted at construction time."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from morel.core.errors import Net


def loopless(adj: sp.spmatrix) -> None:
    """Raise if the adjacency has any self-loops."""
    diag = adj.diagonal()
    if (diag != 0).any():
        raise Net("graph has self-loops")


def symmetric(adj: sp.spmatrix, *, atol: float = 1e-6) -> None:
    """Raise if the adjacency is not symmetric up to ``atol``."""
    diff = adj - adj.T
    if isinstance(diff, sp.spmatrix):
        diff = diff.tocsr()
    if abs(diff).max() > atol:
        raise Net("graph is not symmetric")


def isoless(adj: sp.spmatrix, *, min_degree: int = 1) -> None:
    """Raise if any node has degree below ``min_degree``."""
    degrees = np.asarray(adj.sum(axis=1)).flatten()
    if (degrees < min_degree).any():
        raise Net("graph has isolated nodes")


def finite(adj: sp.spmatrix) -> None:
    """Raise if the adjacency contains non-finite values."""
    if sp.issparse(adj):
        if not np.isfinite(adj.data).all():
            raise Net("graph contains non-finite values")
    elif not np.isfinite(adj).all():
        raise Net("graph contains non-finite values")


def verify_all(adj: sp.spmatrix) -> None:
    """Run the full default invariant suite."""
    finite(adj)
    loopless(adj)
    symmetric(adj)
    isoless(adj)


__all__ = ["finite", "isoless", "loopless", "symmetric", "verify_all"]
