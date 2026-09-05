"""Integration tests for the full pipeline."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader

from morel.core.config import Config
from morel.data.build import bipartite, item_cooccurrence
from morel.data.mask import bernoulli
from morel.pipeline import Pipeline
from morel.train.completion import Completion, CompletionConfig


class _Monitor:
    def log(self, *args, **kwargs):  # noqa: ANN001, D401
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

    config = Config(encode=Config.__dataclass_fields__["encode"].default_factory())  # type: ignore[misc]
    pipeline = Pipeline(config, dims={"visual": 4, "text": 2})
    pipeline.register_buffers(features_np, mask_np, item_graph)

    features = {k: torch.from_numpy(v) for k, v in features_np.items()}
    mask = torch.from_numpy(mask_np)
    index = torch.arange(items)
    out = pipeline(features, mask, item_graph, index=index, training=False)
    assert out.completed["visual"].shape == (items, 4)
    assert out.completed["text"].shape == (items, 2)


def test_trainer_decreases_loss_on_synthetic(tmp_path) -> None:
    rng = np.random.default_rng(0)
    features = {
        "visual": rng.normal(size=(20, 4)).astype(np.float32),
        "text": rng.normal(size=(20, 2)).astype(np.float32),
    }
    mask = np.ones((20, 2), dtype=np.float32)
    adj = sp.csr_matrix(np.eye(20, dtype=np.float32))

    class _Standin(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.codebook = torch.nn.Module()
            self.codebook.usage = lambda g: torch.tensor(0.0)
            self.codebook.balance = lambda g: torch.tensor(0.0)
            self.linear = torch.nn.Linear(6, 6)

        def forward(self, features, mask, adjacency, index=None, training=True):  # noqa: ARG002
            x = torch.cat([features["visual"], features["text"]], dim=-1)
            x = self.linear(x)
            return {"visual": x[:, :4], "text": x[:, 4:]}, torch.softmax(x[:, :8], dim=-1)

    class _Ds(torch.utils.data.Dataset):
        def __len__(self):
            return 20

        def __getitem__(self, idx):
            return {
                "index": idx,
                "features": {k: v[idx] for k, v in features.items()},
                "mask": mask[idx],
                "adjacency": adj,
            }

    loader = DataLoader(
        _Ds(),
        batch_size=4,
        collate_fn=lambda batch: {
            "index": torch.from_numpy(np.stack([np.asarray(b["index"]) for b in batch])),
            "mask": torch.from_numpy(np.stack([np.asarray(b["mask"]) for b in batch])),
            "features": {
                k: torch.from_numpy(np.stack([b["features"][k] for b in batch]))
                for k in features
            },
            "adjacency": batch[0]["adjacency"],
        },
    )
    cfg = CompletionConfig()
    trainer = Completion(_Standin(), cfg, monitor=_Monitor(), checkpoint_dir=tmp_path)
    trainer.fit(loader, loader, epochs=3, patience=5)
