"""Integration test: real Pipeline driven by Completion trainer."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader

from morel.core.config import Config
from morel.core.errors import GraphError
from morel.pipeline import Pipeline
from morel.train.completion import Completion, CompletionConfig
from tests.shared import CompletionDataset, build_path_graph, make_completion_collate


class SilentMonitor:
    """Test monitor that discards metric logs."""

    def log(self, step: int | None = None, **metrics: object) -> None:
        """Discard the call and accept any keyword arguments."""
        return


class Checker:
    """Aggregated test methods for this module."""

    def real(tmp_path) -> None:
        rng = np.random.default_rng(0)
        n = 30
        dim_visual = 6
        dim_text = 4
        features_np = {
            "visual": rng.normal(size=(n, dim_visual)).astype(np.float32),
            "text": rng.normal(size=(n, dim_text)).astype(np.float32),
        }
        mask_np = np.ones((n, 2), dtype=np.float32)
        adj = build_path_graph(n)

        config = Config()
        pipeline = Pipeline(config, dims={"visual": dim_visual, "text": dim_text})
        pipeline.attach(features_np, mask_np, adj)

        loader = DataLoader(
            CompletionDataset(features_np, mask_np, adj),
            batch_size=8,
            collate_fn=make_completion_collate(list(features_np.keys())),
        )
        trainer = Completion(
            pipeline,
            CompletionConfig(),
            monitor=SilentMonitor(),
            checkpoint_dir=tmp_path,
            device="cpu",
        )
        result = trainer.fit(loader, None, epochs=2, patience=4)
        assert "best" in result
        assert torch.isfinite(torch.tensor(result["best"])) or result["best"] == float("inf")

    def pipeline(tmp_path) -> None:
        """Half the items have their text modality masked."""
        rng = np.random.default_rng(1)
        n = 24
        features_np = {
            "visual": rng.normal(size=(n, 4)).astype(np.float32),
            "text": rng.normal(size=(n, 2)).astype(np.float32),
        }
        mask_np = np.ones((n, 2), dtype=np.float32)
        mask_np[n // 2 :, 1] = 0.0
        adj = build_path_graph(n)

        pipeline = Pipeline(Config(), dims={"visual": 4, "text": 2})
        pipeline.attach(features_np, mask_np, adj)

        features_t = {k: torch.from_numpy(v) for k, v in features_np.items()}
        mask_t = torch.from_numpy(mask_np)
        index = torch.arange(n)
        out = pipeline(features_t, mask_t, adj, index=index, training=False)
        assert out.completed["visual"].shape == (n, 4)
        assert out.completed["text"].shape == (n, 2)
        assert torch.isfinite(out.completed["text"]).all()

    def rejects() -> None:
        """Pipeline raises GraphError when an adjacency with self-loops is passed."""
        rng = np.random.default_rng(2)
        n = 8
        features_np = {
            "visual": rng.normal(size=(n, 4)).astype(np.float32),
            "text": rng.normal(size=(n, 2)).astype(np.float32),
        }
        mask_np = np.ones((n, 2), dtype=np.float32)
        bad = sp.csr_matrix(np.eye(n, dtype=np.float32))
        pipeline = Pipeline(Config(), dims={"visual": 4, "text": 2})
        pipeline.attach(features_np, mask_np, bad)

        features_t = {k: torch.from_numpy(v) for k, v in features_np.items()}
        mask_t = torch.from_numpy(mask_np)
        with pytest.raises(GraphError):
            pipeline(features_t, mask_t, bad, training=False)
