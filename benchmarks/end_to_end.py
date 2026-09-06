"""End-to-end training benchmark."""

from __future__ import annotations

import numpy as np

from morel.core.config import Config
from morel.data.build import bipartite as build_bipartite
from morel.data.build import item_cooccurrence
from morel.pipeline import Pipeline
from morel.train.completion import Completion, CompletionConfig


class _Monitor:
    def log(self, *args, **kwargs):
        return None


def bench_epoch_5k(tmp_path, benchmark) -> None:
    rng = np.random.default_rng(0)
    items = 5000
    # Use a real item cooccurrence graph (no self-loops) so the laplacian and
    # retrieval paths exercise the production code path.
    users = max(64, items // 20)
    uids = rng.integers(0, users, size=items * 4)
    iids = rng.integers(0, items, size=items * 4)
    ui = build_bipartite(uids, iids, users, items)
    adjacency = item_cooccurrence(ui)

    features = {
        "visual": rng.normal(size=(items, 8)).astype(np.float32),
        "text": rng.normal(size=(items, 4)).astype(np.float32),
    }
    mask = np.ones((items, 2), dtype=np.float32)

    class _Ds:
        def __len__(self):
            return items

        def __getitem__(self, idx):
            return {
                "index": idx,
                "features": {k: v[idx] for k, v in features.items()},
                "mask": mask[idx],
                "adjacency": adjacency,
            }

    config = Config()
    pipeline = Pipeline(config, dims={"visual": 8, "text": 4})
    pipeline.attach_corpus(features, mask, adjacency)
    trainer = Completion(
        pipeline,
        CompletionConfig(),
        monitor=_Monitor(),
        checkpoint_dir=tmp_path,
    )

    from torch.utils.data import DataLoader

    loader = DataLoader(_Ds(), batch_size=64, collate_fn=_collate)

    def run():
        trainer.run_epoch(loader, 0)

    benchmark(run)


def _collate(batch):  # type: ignore[no-untyped-def]
    import torch

    features_keys = list(batch[0]["features"].keys())
    return {
        "index": torch.from_numpy(np.stack([np.asarray(b["index"]) for b in batch])),
        "mask": torch.from_numpy(np.stack([np.asarray(b["mask"]) for b in batch])),
        "features": {
            k: torch.from_numpy(np.stack([b["features"][k] for b in batch])) for k in features_keys
        },
        "adjacency": batch[0]["adjacency"],
    }
