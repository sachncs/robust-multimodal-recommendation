"""Downstream recommendation models based on LightGCN."""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


class LightGCN(nn.Module):
    """Minimal LightGCN model for downstream recommendation.

    This model learns user and item embeddings via linear message passing
    on a user-item bipartite graph.

    Attributes:
        num_users: Number of users.
        num_items: Number of items.
        num_layers: Number of LightGCN propagation layers.
        user_emb: User embedding table.
        item_emb: Item embedding table.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        """Initialize the LightGCN model.

        Args:
            num_users: Number of users in the dataset.
            num_items: Number of items in the dataset.
            embedding_dim: Dimensionality of user and item embeddings.
            num_layers: Number of graph convolution layers.
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def _build_normalized_adj(
        self, ui_graph: sp.csr_matrix
    ) -> torch.Tensor:
        """Build symmetrically normalized adjacency from a bipartite graph.

        Constructs the full adjacency matrix for the user-item bipartite
        graph and applies symmetric normalization.

        Args:
            ui_graph: Sparse user-item interaction matrix of shape
                (num_users, num_items) in CSR format.

        Returns:
            Sparse COO tensor representing the normalized adjacency
            matrix.
        """
        n_users, n_items = ui_graph.shape
        top = sp.hstack(
            [sp.csr_matrix((n_users, n_users)), ui_graph]
        )
        bottom = sp.hstack(
            [ui_graph.T, sp.csr_matrix((n_items, n_items))]
        )
        adj = sp.vstack([top, bottom])
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv_sqrt = np.zeros_like(rowsum)
        d_inv_sqrt[rowsum > 0] = np.power(
            rowsum[rowsum > 0], -0.5
        )
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        norm_adj = d_inv_sqrt_mat @ adj @ d_inv_sqrt_mat
        norm_adj = sp.coo_matrix(norm_adj)
        indices = torch.from_numpy(
            np.vstack((norm_adj.row, norm_adj.col))
        ).long()
        values = torch.from_numpy(norm_adj.data).float()
        shape = torch.Size(norm_adj.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        ui_graph: sp.spmatrix,
    ) -> torch.Tensor:
        """Compute user-item scores via LightGCN propagation.

        Args:
            users: Tensor of user indices of shape (num_users,).
            items: Tensor of item indices of shape (num_items,).
            ui_graph: Sparse user-item interaction matrix.

        Returns:
            Score matrix of shape (num_users, num_items) where each entry
            represents the predicted affinity between a user and an item.
        """
        adj_norm = self._build_normalized_adj(ui_graph)
        all_emb = torch.cat(
            [self.user_emb.weight, self.item_emb.weight], dim=0
        )
        embs = [all_emb]
        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(adj_norm, all_emb)
            embs.append(all_emb)
        all_emb = torch.stack(embs, dim=0).mean(dim=0)
        u_emb = all_emb[: self.num_users]
        i_emb = all_emb[self.num_users :]
        scores = torch.matmul(u_emb[users], i_emb[items].t())
        return scores
