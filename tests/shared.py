"""Shared test fixtures used across unit, integration, and research tests.

Re-exports the application-layer helpers (``Corpus``,
``build``, ``make_completion_collate``, ...) from
``morel.app.data`` so test code has a single canonical entry point.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

from morel.app.data import (
    Corpus,
    build,
    cast,
    collate,
)


class Monitor:
    """A monitor whose :meth:`log` is a no-op.

    Used as the default monitor in trainer tests so that test runs do not
    write metrics files.
    """

    def log(self, step: int | None = None, **metrics: object) -> None:
        """Discard the call. Accepts any keyword arguments."""
        return


def silent_monitor() -> Monitor:
    """Return a fresh :class:`Monitor`."""
    return Monitor()


def make_completion_collate(features_keys: list[str]) -> Callable:
    """Build a DataLoader collate_fn for completion-stage batches.

    This is a factory because the modality names are only known at
    construction time, not per-batch.

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


def build_path_graph(n: int) -> sp.csr_matrix:
    """Construct a path graph adjacency (no self-loops)."""
    rows: list[int] = []
    cols: list[int] = []
    for i in range(n - 1):
        rows.extend([i, i + 1])
        cols.extend([i + 1, i])
    return sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n, n),
    )


class BPR(Dataset):
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
        """Return the number of batches in the dataset."""
        return self.n_batches

    def __getitem__(self, idx: int) -> dict:
        """Return the generated BPR triple for index ``idx``."""
        rng = np.random.default_rng(idx + self.seed)
        return {
            "users": int(rng.integers(0, self.users)),
            "positive": int(rng.integers(0, self.items)),
            "negative": int(rng.integers(0, self.items)),
        }


def bpr_loader(users: int, items: int, n: int, batch_size: int) -> DataLoader:
    """Build a default DataLoader for BPR tests."""
    return DataLoader(
        BPR(users=users, items=items, n_batches=n),
        batch_size=batch_size,
    )


__all__ = [
    "BPR",
    "Corpus",
    "Monitor",
    "bpr_loader",
    "build",
    "build_path_graph",
    "cast",
    "collate",
    "make_completion_collate",
    "silent_monitor",
]
