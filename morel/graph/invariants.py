"""Graph invariants asserted at construction time."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from morel.core.errors import GraphError


def no_self_loops(adj: sp.spmatrix) -> None:
    """Raise if the adjacency has any self-loops."""
    diag = adj.diagonal()
    if (diag != 0).any():
        raise GraphError("graph has self-loops")


def symmetric(adj: sp.spmatrix, *, atol: float = 1e-6) -> None:
    """Raise if the adjacency is not symmetric up to ``atol``."""
    diff = adj - adj.T
    if isinstance(diff, sp.spmatrix):
        diff = diff.tocsr()
    if abs(diff).max() > atol:
        raise GraphError("graph is not symmetric")


def no_isolated(adj: sp.spmatrix, *, min_degree: int = 1) -> None:
    """Raise if any node has degree below ``min_degree``."""
    degrees = np.asarray(adj.sum(axis=1)).flatten()
    if (degrees < min_degree).any():
        raise GraphError("graph has isolated nodes")


def finite(adj: sp.spmatrix) -> None:
    """Raise if the adjacency contains non-finite values."""
    if sp.issparse(adj):
        if not np.isfinite(adj.data).all():
            raise GraphError("graph contains non-finite values")
    elif not np.isfinite(adj).all():
        raise GraphError("graph contains non-finite values")


def all_invariants(adj: sp.spmatrix) -> None:
    """Run the full default invariant suite."""
    finite(adj)
    no_self_loops(adj)
    symmetric(adj)
    no_isolated(adj)


__all__ = ["all_invariants", "finite", "no_isolated", "no_self_loops", "symmetric"]
