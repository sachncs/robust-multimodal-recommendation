"""Tests for morel.train.completion and recommendation trainers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader

from morel.core.config import Config
from morel.pipeline import Pipeline
from morel.recommend import Light
from morel.train.completion import Completion, CompletionConfig
from morel.train.recommendation import Recommendation, RecommendationConfig
from tests.shared import BPRDataset, build_completion_loader, build_path_graph


def test_completion_trainer_runs_with_real_pipeline(tmp_path: Path, silent_monitor_factory) -> None:
    rng = np.random.default_rng(0)
    n = 20
    dim = 4
    features = {
        "visual": rng.normal(size=(n, dim)).astype(np.float32),
        "text": rng.normal(size=(n, dim)).astype(np.float32),
    }
    mask = np.ones((n, 2), dtype=np.float32)
    adj = build_path_graph(n)

    config = Config()
    model = Pipeline(config, dims={"visual": dim, "text": dim})
    model.attach_corpus(features, mask, adj)

    loader = build_completion_loader(features, mask, adj, batch_size=4)
    cfg = CompletionConfig()
    trainer = Completion(model, cfg, monitor=silent_monitor_factory(), checkpoint_dir=tmp_path)
    trainer.fit(loader, loader, epochs=2, patience=1)
    assert True


def test_recommendation_trainer_runs(tmp_path: Path, silent_monitor_factory) -> None:
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
    model = Light(users=users, items=items, embed=4, layers=2)
    loader = DataLoader(BPRDataset(users, items, n_batches=16), batch_size=4)
    cfg = RecommendationConfig()
    trainer = Recommendation(
        model, cfg, ui_graph=ui, monitor=silent_monitor_factory(), checkpoint_dir=tmp_path
    )
    trainer.fit(loader, loader, epochs=2, patience=1)
    assert True


def test_trainer_honours_config_device(tmp_path: Path, silent_monitor_factory) -> None:
    """Trainer accepts a device argument and resolves via morel.core.device.device()."""
    users, items = 4, 6
    rng = np.random.default_rng(0)
    user_indices = rng.integers(0, users, size=8)
    item_indices = rng.integers(0, items, size=8)
    ui = sp.csr_matrix(
        (np.ones(8, dtype=np.float32), (user_indices, item_indices)),
        shape=(users, items),
    )
    model = Light(users=users, items=items, embed=4, layers=1)
    trainer = Recommendation(
        model,
        RecommendationConfig(),
        ui_graph=ui,
        monitor=silent_monitor_factory(),
        checkpoint_dir=tmp_path,
        device="cpu",
    )
    assert trainer.device == torch.device("cpu")


def test_trainer_amp_cpu_runs(tmp_path: Path, silent_monitor_factory) -> None:
    """AMP=true on CPU must not crash; AMP path is exercised without device promotion."""
    users, items = 4, 6
    rng = np.random.default_rng(0)
    user_indices = rng.integers(0, users, size=8)
    item_indices = rng.integers(0, items, size=8)
    ui = sp.csr_matrix(
        (np.ones(8, dtype=np.float32), (user_indices, item_indices)),
        shape=(users, items),
    )
    model = Light(users=users, items=items, embed=4, layers=1)
    trainer = Recommendation(
        model,
        RecommendationConfig(),
        ui_graph=ui,
        monitor=silent_monitor_factory(),
        checkpoint_dir=tmp_path,
        device="cpu",
        amp=True,
    )
    loader = DataLoader(BPRDataset(users, items, n_batches=4), batch_size=2)
    trainer.fit(loader, None, epochs=1, patience=2)
    assert trainer.scaler is not None
