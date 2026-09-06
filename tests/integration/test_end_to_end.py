"""Integration tests for the full pipeline."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader

from tests.shared import CompletionDataset, build_path_graph, make_completion_collate
from morel.core.config import Config
from morel.data.build import bipartite, item_cooccurrence
from morel.data.mask import bernoulli
from morel.pipeline import Pipeline
from morel.train.completion import Completion, CompletionConfig


class SilentMonitor:
    def log(self, step: int | None = None, **metrics: object) -> None:
        return None


def test_end_to_end_synthetic(tmp_path) -> None:
    rng = np.random.default_rng(0)
    users, items = 30, 50
    uids = rng.integers(0, users, size=300)
    iids = rng.integers(0, items, size=300)
    ui = bipartite(uids, iids, users, items)
    item_graph = item_cooccurrence(ui)

    features_np = {
        "visual": rng.normal(size=(items, 4)).astype(np.float32),
        "text": rng.normal(size=(items, 2)).astype(np.float32),
    }
    mask_np = bernoulli(items, 2, 0.4, seed=0).to_numpy()

    import torch

    config = Config()
    pipeline = Pipeline(config, dims={"visual": 4, "text": 2})
    pipeline.attach_corpus(features_np, mask_np, item_graph)

    features = {k: torch.from_numpy(v) for k, v in features_np.items()}
    mask = torch.from_numpy(mask_np)
    index = torch.arange(items)
    out = pipeline(features, mask, item_graph, index=index, training=False)
    assert out.completed["visual"].shape == (items, 4)
    assert out.completed["text"].shape == (items, 2)


def test_trainer_decreases_loss_on_synthetic(tmp_path) -> None:
    rng = np.random.default_rng(0)
    n = 20
    features = {
        "visual": rng.normal(size=(n, 4)).astype(np.float32),
        "text": rng.normal(size=(n, 2)).astype(np.float32),
    }
    mask = np.ones((n, 2), dtype=np.float32)
    adj = build_path_graph(n)

    dataset = CompletionDataset(features, mask, adj)
    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=make_completion_collate(list(features.keys())),
    )
    cfg = CompletionConfig()
    pipeline = Pipeline(Config(), dims={"visual": 4, "text": 2})
    pipeline.attach_corpus(features, mask, adj)
    trainer = Completion(
        pipeline, cfg, monitor=SilentMonitor(), checkpoint_dir=tmp_path
    )
    trainer.fit(loader, loader, epochs=3, patience=5)
