"""Shared data helpers for the application services.

Helpers that the Experiment service (and the tests) both need live here so
neither layer has to redeclare them.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset


class CompletionDataset(Dataset):
    """In-memory completion-stage dataset.

    Each item returns ``{index, features, mask, adjacency}`` shaped for the
    completion trainer's collate function.
    """

    def __init__(
        self,
        features: dict[str, np.ndarray],
        mask: np.ndarray,
        adjacency: sp.csr_matrix,
    ) -> None:
        self.features = features
        self.mask = mask
        self.adjacency = adjacency
        self.n = mask.shape[0]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return {
            "index": idx,
            "features": {k: v[idx] for k, v in self.features.items()},
            "mask": self.mask[idx],
            "adjacency": self.adjacency,
        }


def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Wrap :func:`torch.from_numpy` for type convenience."""
    return torch.from_numpy(array)


def collate_completion(batch: list[dict]) -> dict:
    """Collate a list of completion-stage samples into a torch dict."""
    features_keys = list(batch[0]["features"].keys())
    return {
        "index": torch.from_numpy(np.stack([np.asarray(b["index"]) for b in batch])),
        "mask": torch.from_numpy(np.stack([np.asarray(b["mask"]) for b in batch])),
        "features": {
            k: torch.from_numpy(np.stack([b["features"][k] for b in batch])) for k in features_keys
        },
        "adjacency": batch[0]["adjacency"],
    }


def build_completion_loader(
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    adjacency: sp.csr_matrix,
    batch_size: int = 8,
) -> DataLoader:
    """Default DataLoader for the completion stage."""
    return DataLoader(
        CompletionDataset(features, mask, adjacency),
        batch_size=batch_size,
        collate_fn=collate_completion,
    )


def synth_bipartite(
    rng: np.random.Generator, items: int, users: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one random (user_ids, item_ids) sample.

    Used by the experiment's synthetic dataset builder.
    """
    pairs = (
        rng.integers(0, users, size=items * 5),
        rng.integers(0, items, size=items * 5),
    )
    return pairs


__all__ = [
    "CompletionDataset",
    "build_completion_loader",
    "collate_completion",
    "numpy_to_tensor",
    "synth_bipartite",
]
