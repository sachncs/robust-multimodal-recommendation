"""Recommender Protocol.

Defines the contract that any downstream ranker must satisfy so the
``Recommendation`` trainer and the serve stack can consume them
uniformly.
"""

from __future__ import annotations

from typing import Protocol

import scipy.sparse as sp
import torch


class Recommender(Protocol):
    """One downstream ranker.

    Implementations accept a batch of user ids, a batch of item ids, and
    optionally a bipartite ``ui_graph`` whose normalized adjacency they
    may rebuild. They return a ``(B_u, B_i)`` score matrix.
    """

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        ui_graph: sp.csr_matrix | None = None,
    ) -> torch.Tensor:  # pragma: no cover - protocol
        ...


__all__ = ["Recommender"]
