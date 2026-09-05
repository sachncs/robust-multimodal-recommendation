"""Tests for morel.train.completion and recommendation trainers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from morel.train.completion import Completion, CompletionConfig
from morel.train.recommendation import Recommendation, RecommendationConfig


class _FeatureDataset(Dataset):
    def __init__(self, features: dict[str, np.ndarray], mask: np.ndarray, adjacency: sp.csr_matrix) -> None:
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


class _NoOpMonitor:
    def log(self, *args, **kwargs):  # noqa: ANN001, D401
        return None


class _Standin(nn.Module):
    """Tiny module that emits (preds, probs) with matching shapes for the trainer."""

    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.codebook = nn.Module()
        self.codebook.usage = lambda g: torch.tensor(0.0)
        self.codebook.balance = lambda g: torch.tensor(0.0)
        self.linear = nn.Linear(dim * 2, dim * 2)

    def forward(self, features, mask, adjacency, index=None, training=True):  # noqa: ARG002
        x = torch.cat([features["visual"], features["text"]], dim=-1)
        x = self.linear(x)
        # Split back into two modalities of equal dim.
        half = x.shape[-1] // 2
        v = x[:, :half]
        t = x[:, half:]
        return {"visual": v, "text": t}, torch.softmax(x[:, :max(half, 2)], dim=-1)


def test_completion_trainer_runs(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    dim = 4
    features = {
        "visual": rng.normal(size=(20, dim)).astype(np.float32),
        "text": rng.normal(size=(20, dim)).astype(np.float32),
    }
    mask = np.ones((20, 2), dtype=np.float32)
    adj = sp.csr_matrix(np.eye(20, dtype=np.float32))

    model = _Standin(dim=dim)
    dataset = _FeatureDataset(features, mask, adj)
    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=lambda batch: {
            "index": torch.from_numpy(np.stack([np.asarray(b["index"]) for b in batch])),
            "mask": torch.from_numpy(np.stack([np.asarray(b["mask"]) for b in batch])),
            "features": {
                k: torch.from_numpy(np.stack([b["features"][k] for b in batch]))
                for k in features.keys()
            },
            "adjacency": batch[0]["adjacency"],
        },
    )
    cfg = CompletionConfig()
    trainer = Completion(model, cfg, monitor=_NoOpMonitor(), checkpoint_dir=tmp_path)
    trainer.fit(loader, loader, epochs=2, patience=1)
    assert True


def test_recommendation_trainer_runs(tmp_path: Path) -> None:
    users, items = 8, 12
    rng = np.random.default_rng(0)
    user_indices = rng.integers(0, users, size=64)
    item_indices = rng.integers(0, items, size=64)
    ui = sp.csr_matrix(
        (
            np.ones(64, dtype=np.float32),
            (user_indices, item_indices),
        ),
        shape=(users, items),
    )

    class _Ds(Dataset):
        def __len__(self) -> int:
            return 16

        def __getitem__(self, idx: int) -> dict:
            return {
                "users": int(np.random.default_rng(idx).integers(0, users)),
                "positive": int(np.random.default_rng(idx + 1000).integers(0, items)),
                "negative": int(np.random.default_rng(idx + 2000).integers(0, items)),
            }

    from morel.recommend import Light

    model = Light(users=users, items=items, embed=4, layers=2)
    loader = DataLoader(_Ds(), batch_size=4)
    cfg = RecommendationConfig()
    trainer = Recommendation(model, cfg, ui_graph=ui, monitor=_NoOpMonitor(), checkpoint_dir=tmp_path)
    trainer.fit(loader, loader, epochs=2, patience=1)
    assert True
