"""Positional encoding modules for graph structures."""


import scipy.sparse as sp
import torch
import torch.nn as nn

from rmr.models.utils import compute_laplacian_pe


class LaplacianPE(nn.Module):
    """Compute and cache Laplacian positional encodings.

    This module precomputes the bottom-k eigenvectors of the normalized
    graph Laplacian and caches the result by the id of the input
    adjacency matrix.

    Attributes:
        k: Number of eigenvectors to compute.
    """

    def __init__(self, k: int = 20) -> None:
        """Initialize the LaplacianPE module.

        Args:
            k: Number of nontrivial eigenvectors to compute.
        """
        super().__init__()
        self.k = k
        self._cache: dict[int, torch.Tensor] = {}

    def forward(self, adjacency: sp.spmatrix) -> torch.Tensor:
        """Compute or retrieve cached Laplacian positional encodings.

        Args:
            adjacency: Sparse adjacency matrix.

        Returns:
            Tensor of shape (n_nodes, k) containing the Laplacian
            positional encodings.
        """
        key = id(adjacency)
        if key not in self._cache:
            pe = compute_laplacian_pe(adjacency, k=self.k)
            self._cache[key] = torch.from_numpy(pe).float()
        return self._cache[key]

    def clear_cache(self) -> None:
        """Clear the internal adjacency-to-PE cache."""
        self._cache.clear()
