"""PyTorch Dataset definition for GRE-MC graph training."""

from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


class GREMCGraphDataset(Dataset):
    """Dataset for GRE-MC training.

    Returns per-item features, modality mask, and index.
    """

    def __init__(
        self,
        features: dict[str, np.ndarray],
        mask: np.ndarray,
        adjacency: sp.spmatrix,
    ) -> None:
        """Initialize the dataset.

        Args:
            features: Dictionary mapping modality names to NumPy arrays.
            mask: Binary mask array of shape (n_items, n_modalities).
            adjacency: Sparse adjacency matrix for the item graph.
        """
        self.features = {
            k: torch.from_numpy(v) for k, v in features.items()
        }
        self.mask = torch.from_numpy(mask)
        self.adjacency = adjacency
        self.n_items = mask.shape[0]

    def __len__(self) -> int:
        """Return the number of items in the dataset.

        Returns:
            The number of items.
        """
        return self.n_items

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Retrieve a single sample by index.

        Args:
            idx: Item index.

        Returns:
            A dictionary with keys ``"index"``, ``"features"``, and
            ``"mask"``.
        """
        feats = {k: v[idx] for k, v in self.features.items()}
        m = self.mask[idx]
        return {
            "index": idx,
            "features": feats,
            "mask": m,
        }
