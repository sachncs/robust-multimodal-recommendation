"""End-to-end training benchmark."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from morel.core.config import Config
from morel.pipeline import Pipeline
from morel.train.completion import Completion, CompletionConfig


class _Monitor:
    def log(self, *args, **kwargs):  # noqa: ANN001, D401
        return None


def bench_epoch_5k(tmp_path, benchmark) -> None:
    rng = np.random.default_rng(0)
    items = 5000
    features = {
        "visual": rng.normal(size=(items, 8)).astype(np.float32),
        "text": rng.normal(size=(items, 4)).astype(np.float32),
    }
    mask = np.ones((items, 2), dtype=np.float32)
    adjacency = sp.csr_matrix(np.eye(items, dtype=np.float32))

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
    trainer = Completion(
        pipeline,
        CompletionConfig(),
        monitor=_Monitor(),
        checkpoint_dir=tmp_path,
    )

    from torch.utils.data import DataLoader

    loader = DataLoader(_Ds(), batch_size=64)

    def run():
        trainer._run_epoch(loader, 0)

    benchmark(run)
