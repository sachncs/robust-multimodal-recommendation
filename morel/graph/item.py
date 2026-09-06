"""Item-item co-occurrence graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from morel.graph import invariants
from morel.graph.bipartite import Bipartite


@dataclass(frozen=True)
class Item:
    """Symmetric item-item co-occurrence graph.

    Invariants enforced on construction:
        - no self-loops
        - symmetric (binary co-occurrence)
        - finite values
    """

    matrix: sp.csr_matrix

    def __post_init__(self) -> None:
        """Enforce graph invariants on construction."""
        invariants.no_loops(self.matrix)
        invariants.symmetric(self.matrix)
        invariants.finite(self.matrix)

    @classmethod
    def from_bip(cls, bipartite: Bipartite) -> Item:
        """Construct the item-item graph from a bipartite user-item graph."""
        cooc = bipartite.matrix.T @ bipartite.matrix
        cooc = cooc.sign()
        cooc.setdiag(0)
        cooc.eliminate_zeros()
        return cls(matrix=cooc.tocsr())

    @property
    def items(self) -> int:
        """Return the number of items."""
        return int(self.matrix.shape[0])

    @property
    def edges(self) -> int:
        """Return the number of edges."""
        return int(self.matrix.nnz)

    def adjacency(self) -> sp.csr_matrix:
        """Return the underlying sparse adjacency."""
        return self.matrix

    def neighbors(self, item: int) -> np.ndarray:
        """Return the sorted list of item indices connected to ``item``."""
        row_start = self.matrix.indptr[item]
        row_end = self.matrix.indptr[item + 1]
        return np.sort(self.matrix.indices[row_start:row_end])

    def __getstate__(self) -> dict[str, Any]:
        """Pickle support."""
        return {
            "data": self.matrix.data,
            "indices": self.matrix.indices,
            "indptr": self.matrix.indptr,
            "shape": self.matrix.shape,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Pickle support."""
        object.__setattr__(
            self,
            "matrix",
            sp.csr_matrix(
                (state["data"], state["indices"], state["indptr"]),
                shape=state["shape"],
            ),
        )


__all__ = ["Item"]
