"""End-to-end training benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

from morel.core.config import Config
from morel.data.build import bipartite, item_cooccurrence
from morel.pipeline import Pipeline
from morel.train.completion import Completion, CompletionConfig


class Monitor:
    """No-op monitor stub used by the trainer interface in benchmarks."""

    def log(self, *args: Any, **kwargs: Any) -> None:
        """Accept any log call without effect."""


class Corpus(Dataset[dict[str, Any]]):
    """In-memory corpus that materialises one item row per ``__getitem__``."""

    def __init__(self, items: int, features: dict[str, np.ndarray],
                 mask: np.ndarray, adjacency: sp.csr_matrix) -> None:
        self.items = items
        self.features = features
        self.mask = mask
        self.adjacency = adjacency

    def __len__(self) -> int:
        return self.items

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "index": idx,
            "features": {k: v[idx] for k, v in self.features.items()},
            "mask": self.mask[idx],
            "adjacency": self.adjacency,
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack a list of corpus rows into a single training batch."""
    keys = list(batch[0]["features"].keys())
    return {
        "index": torch.from_numpy(np.stack([np.asarray(b["index"]) for b in batch])),
        "mask": torch.from_numpy(np.stack([np.asarray(b["mask"]) for b in batch])),
        "features": {
            k: torch.from_numpy(np.stack([b["features"][k] for b in batch]))
            for k in keys
        },
        "adjacency": batch[0]["adjacency"],
    }


def bench_5k(tmp_path: Any, benchmark: Any) -> None:
    """Measure one training epoch on a 5k-item synthetic corpus."""
    rng = np.random.default_rng(0)
    items = 5000
    users = max(64, items // 20)
    uids = rng.integers(0, users, size=items * 4)
    iids = rng.integers(0, items, size=items * 4)
    ui = bipartite(uids, iids, users, items)
    adjacency = item_cooccurrence(ui)
    features = {
        "visual": rng.normal(size=(items, 8)).astype(np.float32),
        "text": rng.normal(size=(items, 4)).astype(np.float32),
    }
    mask = np.ones((items, 2), dtype=np.float32)

    config = Config()
    pipeline = Pipeline(config, dims={"visual": 8, "text": 4})
    pipeline.attach(features, mask, adjacency)
    trainer = Completion(
        pipeline,
        CompletionConfig(),
        monitor=Monitor(),
        checkpoint_dir=tmp_path,
    )
    loader = DataLoader(Corpus(items, features, mask, adjacency),
                        batch_size=64, collate_fn=collate)

    def run() -> None:
        trainer.run(loader, 0)

    benchmark(run)
