"""Tests for morel.train.completion and recommendation trainers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from morel.core.config import Config
from morel.pipeline import Pipeline
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


def _build_path_graph(n: int) -> sp.csr_matrix:
    """Build a simple path graph adjacency without self-loops."""
    arr = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        arr[i, i + 1] = 1
        arr[i + 1, i] = 1
    return sp.csr_matrix(arr)


def _collate_features(batch, features):
    return {
        "index": torch.from_numpy(np.stack([np.asarray(b["index"]) for b in batch])),
        "mask": torch.from_numpy(np.stack([np.asarray(b["mask"]) for b in batch])),
        "features": {
            k: torch.from_numpy(np.stack([b["features"][k] for b in batch]))
            for k in features.keys()
        },
        "adjacency": batch[0]["adjacency"],
    }


def test_completion_trainer_runs_with_real_pipeline(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 20
    dim = 4
    features = {
        "visual": rng.normal(size=(n, dim)).astype(np.float32),
        "text": rng.normal(size=(n, dim)).astype(np.float32),
    }
    mask = np.ones((n, 2), dtype=np.float32)
    adj = _build_path_graph(n)

    config = Config()
    model = Pipeline(config, dims={"visual": dim, "text": dim})
    model.attach_corpus(features, mask, adj)

    dataset = _FeatureDataset(features, mask, adj)
    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=lambda batch: _collate_features(batch, features),
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


def test_trainer_honours_config_device(tmp_path: Path) -> None:
    """Trainer accepts a device argument and resolves via morel.core.device.device()."""
    from morel.recommend import Light

    users, items = 4, 6
    rng = np.random.default_rng(0)
    user_indices = rng.integers(0, users, size=8)
    item_indices = rng.integers(0, items, size=8)
    ui = sp.csr_matrix(
        (np.ones(8, dtype=np.float32), (user_indices, item_indices)),
        shape=(users, items),
    )

    class _Ds(Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, idx: int) -> dict:
            return {
                "users": idx % users,
                "positive": idx % items,
                "negative": (idx + 1) % items,
            }

    model = Light(users=users, items=items, embed=4, layers=1)
    trainer = Recommendation(
        model,
        RecommendationConfig(),
        ui_graph=ui,
        monitor=_NoOpMonitor(),
        checkpoint_dir=tmp_path,
        device="cpu",
    )
    assert trainer.device == torch.device("cpu")


def test_trainer_amp_cpu_runs(tmp_path: Path) -> None:
    """AMP=true on CPU must not crash; AMP path is exercised without device promotion."""
    from morel.recommend import Light

    users, items = 4, 6
    rng = np.random.default_rng(0)
    user_indices = rng.integers(0, users, size=8)
    item_indices = rng.integers(0, items, size=8)
    ui = sp.csr_matrix(
        (np.ones(8, dtype=np.float32), (user_indices, item_indices)),
        shape=(users, items),
    )

    class _Ds(Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, idx: int) -> dict:
            return {
                "users": idx % users,
                "positive": idx % items,
                "negative": (idx + 1) % items,
            }

    model = Light(users=users, items=items, embed=4, layers=1)
    trainer = Recommendation(
        model,
        RecommendationConfig(),
        ui_graph=ui,
        monitor=_NoOpMonitor(),
        checkpoint_dir=tmp_path,
        device="cpu",
        amp=True,
    )
    loader = DataLoader(_Ds(), batch_size=2)
    trainer.fit(loader, None, epochs=1, patience=2)
    assert trainer._scaler is not None
