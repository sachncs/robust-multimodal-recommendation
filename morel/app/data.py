"""Shared data helpers for the application services.

Helpers that the Experiment service (and the tests) both need live here so
neither layer has to redeclare them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from morel.core.errors import DataError
from morel.recommend.bpr import ranks_to_items


class Corpus(Dataset[dict[str, Any]]):
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
        """Return the number of samples."""
        return int(self.n)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single sample dict for the given index."""
        return {
            "index": idx,
            "features": {k: v[idx] for k, v in self.features.items()},
            "mask": self.mask[idx],
            "adjacency": self.adjacency,
        }


def to_tensor(array: np.ndarray) -> torch.Tensor:
    """Wrap :func:`torch.from_numpy` for type convenience."""
    return torch.from_numpy(array)


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
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


def split(
    dataset: Dataset[dict[str, Any]], *, val_fraction: float, seed: int
) -> tuple[Dataset[dict[str, Any]], Dataset[dict[str, Any]] | None]:
    """Split ``dataset`` deterministically into train and validation parts.

    Args:
        dataset: A sized dataset.
        val_fraction: Fraction held out for validation, in ``[0, 1)``.
        seed: Seed for the permutation, so the split is reproducible.

    Returns
    -------
        ``(train, val)``. ``val`` is ``None`` when the fraction rounds to no
        samples or would consume the whole dataset, in which case training
        proceeds without validation rather than on an empty split.

    Raises
    ------
        DataError: If ``val_fraction`` is outside ``[0, 1)``.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise DataError(f"validation fraction must be in [0, 1), got {val_fraction}")
    total = len(dataset)  # type: ignore[arg-type]  # Dataset has no Sized bound
    held_out = round(total * val_fraction)
    if held_out == 0 or held_out >= total:
        return dataset, None
    order = np.random.default_rng(seed).permutation(total)
    val_indices = sorted(int(i) for i in order[:held_out])
    train_indices = sorted(int(i) for i in order[held_out:])
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def build_loaders(
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    adjacency: sp.csr_matrix,
    *,
    batch_size: int = 8,
    val_fraction: float = 0.0,
    seed: int = 0,
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]] | None]:
    """Build the completion train loader and, if requested, a validation loader."""
    train, val = split(
        Corpus(features, mask, adjacency), val_fraction=val_fraction, seed=seed
    )
    train_loader = DataLoader(train, batch_size=batch_size, collate_fn=collate)
    if val is None:
        return train_loader, None
    return train_loader, DataLoader(val, batch_size=batch_size, collate_fn=collate)


def build_loader(
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    adjacency: sp.csr_matrix,
    batch_size: int = 8,
) -> DataLoader[dict[str, Any]]:
    """Default DataLoader for the completion stage."""
    return DataLoader(
        Corpus(features, mask, adjacency),
        batch_size=batch_size,
        collate_fn=collate,
    )


class BPR(Dataset[dict[str, Any]]):
    """BPR triples drawn from a real interaction matrix.

    Each item is a ``(user, positive, negative)`` triple where the positive is
    an item the user actually interacted with and the negative is one they did
    not. Users with no interactions are skipped, since a BPR triple is
    undefined for them.

    Indexing is a pure function of the index and the seed, so the epoch a
    sample belongs to does not change it and the dataset is reproducible
    across processes and workers.
    """

    def __init__(self, ui_graph: sp.csr_matrix, *, length: int, seed: int = 0) -> None:
        """Bind the interaction matrix and precompute each user's positives.

        Args:
            ui_graph: ``(users, items)`` interaction matrix.
            length: Number of triples one epoch draws.
            seed: Base seed; sample ``i`` is derived from ``seed + i``.

        Raises
        ------
            DataError: If no user has any interaction, or if any user has no
                negative available.
        """
        self.items = int(ui_graph.shape[1])
        self.length = int(length)
        self.seed = int(seed)
        indptr, indices = ui_graph.indptr, ui_graph.indices
        self.positives = {
            user: np.unique(indices[indptr[user] : indptr[user + 1]])
            for user in range(int(ui_graph.shape[0]))
            if indptr[user + 1] > indptr[user]
        }
        if not self.positives:
            raise DataError("no user has any interaction; cannot form BPR triples")
        self.users = sorted(self.positives)
        crowded = [u for u, pos in self.positives.items() if pos.size >= self.items]
        if crowded:
            raise DataError(
                f"user {crowded[0]} interacts with every one of {self.items} items; "
                "no negative can be sampled"
            )

    def __len__(self) -> int:
        """Return the number of triples in one epoch."""
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return the BPR triple for ``idx``, sampled deterministically."""
        rng = np.random.default_rng(self.seed + idx)
        user = int(self.users[int(rng.integers(0, len(self.users)))])
        positives = self.positives[user]
        positive = int(positives[int(rng.integers(0, positives.size))])
        rank = int(rng.integers(0, self.items - positives.size))
        negative = int(ranks_to_items(np.array([rank], dtype=np.int64), positives)[0])
        return {"users": user, "positive": positive, "negative": negative}


def recommend_loader(
    ui_graph: sp.csr_matrix,
    *,
    batch_size: int = 1024,
    length: int | None = None,
    seed: int = 0,
) -> DataLoader[dict[str, Any]]:
    """Build the BPR DataLoader for the recommendation stage.

    Args:
        ui_graph: ``(users, items)`` interaction matrix.
        batch_size: Triples per batch.
        length: Triples per epoch; defaults to the number of interactions.
        seed: Base seed for deterministic sampling.
    """
    if length is None:
        length = max(int(ui_graph.nnz), 1)
    return DataLoader(
        BPR(ui_graph, length=length, seed=seed),
        batch_size=batch_size,
    )


def recommend_loaders(
    ui_graph: sp.csr_matrix,
    *,
    batch_size: int = 1024,
    length: int | None = None,
    val_fraction: float = 0.0,
    seed: int = 0,
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]] | None]:
    """Build the BPR train loader and, if requested, a validation loader."""
    if length is None:
        length = max(int(ui_graph.nnz), 1)
    train, val = split(
        BPR(ui_graph, length=length, seed=seed), val_fraction=val_fraction, seed=seed
    )
    train_loader = DataLoader(train, batch_size=batch_size)
    if val is None:
        return train_loader, None
    return train_loader, DataLoader(val, batch_size=batch_size)


def synth_bipartite(
    rng: np.random.Generator, items: int, users: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one random (user_ids, item_ids) sample.

    Used by the experiment's synthetic dataset builder.
    """
    return (
        rng.integers(0, users, size=items * 5),
        rng.integers(0, items, size=items * 5),
    )


__all__ = [
    "BPR",
    "Corpus",
    "build_loader",
    "build_loaders",
    "collate",
    "recommend_loader",
    "recommend_loaders",
    "split",
    "synth_bipartite",
    "to_tensor",
]
