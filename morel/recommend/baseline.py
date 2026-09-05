"""Baseline recommenders: matrix factorization and popularity."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


class MF(nn.Module):
    """Matrix factorization with dot-product scoring."""

    def __init__(self, users: int, items: int, *, embed: int = 64) -> None:
        super().__init__()
        if users <= 0 or items <= 0:
            raise ValueError("users and items must be positive")
        self.users = users
        self.items = items
        self.user_emb = nn.Embedding(users, embed)
        self.item_emb = nn.Embedding(items, embed)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        ui_graph: sp.csr_matrix | None = None,
    ) -> torch.Tensor:
        """Score users against items via dot product.

        Args:
            users: ``(B_u,)`` long tensor.
            items: ``(B_i,)`` long tensor.
            ui_graph: Unused; accepted for Protocol compatibility.

        Returns:
            ``(B_u, B_i)`` scores.
        """
        return self.user_emb(users) @ self.item_emb(items).t()


class Pop(nn.Module):
    """Popularity baseline: scores proportional to item interaction count."""

    def __init__(self, users: int, items: int) -> None:
        super().__init__()
        if users <= 0 or items <= 0:
            raise ValueError("users and items must be positive")
        self.users = users
        self.items = items
        self.register_buffer("popularity", torch.zeros(items))
        self._trained = False

    def fit(self, ui: sp.csr_matrix) -> None:
        """Precompute item popularity from the interaction matrix."""
        counts = np.asarray(ui.sum(axis=0)).flatten().astype(np.float32)
        self.popularity = torch.from_numpy(counts)
        self._trained = True

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        ui_graph: sp.csr_matrix | None = None,
    ) -> torch.Tensor:
        """Score users against items via popularity broadcasted to users."""
        if not self._trained:
            if ui_graph is None:
                raise RuntimeError("Pop must be fitted with a ui_graph first")
            self.fit(ui_graph)
        scores = self.popularity.unsqueeze(0).expand(users.shape[0], -1)
        return scores[:, items]


__all__ = ["MF", "Pop"]
