"""Bipartite user-item graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class Bipartite:
    """Immutable user-item bipartite graph."""

    matrix: sp.csr_matrix

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            raise ValueError("bipartite matrix must be 2-D")
        if not sp.issparse(self.matrix):
            raise ValueError("bipartite matrix must be sparse")

    @property
    def users(self) -> int:
        """Return the number of users."""
        return int(self.matrix.shape[0])

    @property
    def items(self) -> int:
        """Return the number of items."""
        return int(self.matrix.shape[1])

    @property
    def edges(self) -> int:
        """Return the number of nonzero entries."""
        return int(self.matrix.nnz)

    def adjacency(self) -> sp.csr_matrix:
        """Return the underlying sparse adjacency."""
        return self.matrix

    def to_dense(self) -> np.ndarray:
        """Return a dense copy as a numpy array."""
        return self.matrix.toarray()

    def __getstate__(self) -> dict[str, Any]:
        """Pickle support: return CSR arrays."""
        return {
            "data": self.matrix.data,
            "indices": self.matrix.indices,
            "indptr": self.matrix.indptr,
            "shape": self.matrix.shape,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Pickle support: rebuild from CSR arrays."""
        object.__setattr__(
            self,
            "matrix",
            sp.csr_matrix(
                (state["data"], state["indices"], state["indptr"]),
                shape=state["shape"],
            ),
        )


__all__ = ["Bipartite"]
