"""Shared test fixtures used across unit, integration, and research tests.

Each public name communicates its role; no ``_foo`` or ``__foo`` markers.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset


def build_path_graph(n: int) -> sp.csr_matrix:
    """Construct a path graph adjacency (no self-loops).

    Args
    ----
    n : int
        Number of nodes.

    Returns
    -------
    sp.csr_matrix
        ``(n, n)`` adjacency with edges between consecutive nodes.
    """
    rows: list[int] = []
    cols: list[int] = []
    for i in range(n - 1):
        rows.extend([i, i + 1])
        cols.extend([i + 1, i])
    return sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n, n),
    )


def silent_monitor() -> "SilentMonitor":
    """A monitor whose .log() is a no-op. Used in trainer tests."""
    return SilentMonitor()


def make_completion_collate(features_keys: list[str]) -> Callable:
    """Build a DataLoader collate_fn for completion-stage batches.

    Args
    ----
    features_keys : list[str]
        Modality names whose ``(B, d_m)`` feature tensors are batched.

    Returns
    -------
    Callable
        A function suitable for ``DataLoader(collate_fn=...)``.
    """

    def collate(batch: list[dict]) -> dict:
        return {
            "index": torch.from_numpy(np.stack([np.asarray(b["index"]) for b in batch])),
            "mask": torch.from_numpy(np.stack([np.asarray(b["mask"]) for b in batch])),
            "features": {
                k: torch.from_numpy(np.stack([b["features"][k] for b in batch]))
                for k in features_keys
            },
            "adjacency": batch[0]["adjacency"],
        }

    return collate


class SilentMonitor:
    """A monitor whose .log() is a no-op."""

    def log(self, *args, **kwargs):  # noqa: ANN001, D401
        return None


class CompletionDataset(Dataset):
    """In-memory completion-stage dataset.

    Each item returns ``{index, features, mask, adjacency}`` shaped
    for the completion trainer's collate function.
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


class BPRDataset(Dataset):
    """In-memory BPR dataset yielding ``(users, positive, negative)``.

    The seed controls the deterministic generator; the index controls
    which generated triple is returned. This guarantees reproducible
    training data across test runs.
    """

    def __init__(
        self,
        users: int,
        items: int,
        n_batches: int,
        seed: int = 0,
    ) -> None:
        self.users = users
        self.items = items
        self.n_batches = n_batches
        self.seed = seed

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(idx + self.seed)
        return {
            "users": int(rng.integers(0, self.users)),
            "positive": int(rng.integers(0, self.items)),
            "negative": int(rng.integers(0, self.items)),
        }


def build_bpr_loader(users: int, items: int, n: int, batch_size: int) -> DataLoader:
    """Build a default DataLoader for BPR tests."""
    return DataLoader(
        BPRDataset(users=users, items=items, n_batches=n),
        batch_size=batch_size,
    )


def build_completion_loader(
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    adjacency: sp.csr_matrix,
    batch_size: int = 4,
) -> DataLoader:
    """Build a default DataLoader for completion tests."""
    return DataLoader(
        CompletionDataset(features, mask, adjacency),
        batch_size=batch_size,
        collate_fn=make_completion_collate(list(features.keys())),
    )


__all__ = [
    "BPRDataset",
    "CompletionDataset",
    "SilentMonitor",
    "build_bpr_loader",
    "build_completion_loader",
    "build_path_graph",
    "make_completion_collate",
    "silent_monitor",
]
