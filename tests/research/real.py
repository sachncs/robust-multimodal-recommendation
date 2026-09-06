"""Real-Pipeline research validation: end-to-end completion training."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

from morel.core.config import Config
from morel.core.seed import seed as seed_everything
from morel.pipeline import Pipeline
from morel.train.completion import Completion, Fit


class Monitor:
    """Test monitor that discards metric logs."""

    def log(self, step: int | None = None, **metrics: object) -> None:
        """Discard the call and accept any keyword arguments."""
        return


def build(n: int) -> sp.csr_matrix:
    rows, cols = [], []
    for i in range(n - 1):
        rows.extend([i, i + 1])
        cols.extend([i + 1, i])
    return sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n, n),
    )


class Checker:
    """Aggregated test methods for this module."""

    def real(self) -> None:
        seed_everything(0)
        rng = np.random.default_rng(0)
        n = 60
        features_np = {
            "visual": rng.normal(size=(n, 8)).astype(np.float32),
            "text": rng.normal(size=(n, 4)).astype(np.float32),
        }
        mask_np = np.ones((n, 2), dtype=np.float32)
        adj = build(n)

        config = Config()
        pipeline = Pipeline(config, dims={"visual": 8, "text": 4})
        pipeline.attach(features_np, mask_np, adj)

        class Batch(Dataset):
            def __len__(self) -> int:
                return n

            def __getitem__(self, idx: int) -> dict:
                return {
                    "index": idx,
                    "features": {k: v[idx] for k, v in features_np.items()},
                    "mask": mask_np[idx],
                    "adjacency": adj,
                }

        loader = DataLoader(
            Batch(),
            batch_size=16,
            collate_fn=lambda batch: {
                "index": torch.from_numpy(np.stack([np.asarray(b["index"]) for b in batch])),
                "mask": torch.from_numpy(np.stack([np.asarray(b["mask"]) for b in batch])),
                "features": {
                    k: torch.from_numpy(np.stack([b["features"][k] for b in batch]))
                    for k in features_np
                },
                "adjacency": batch[0]["adjacency"],
            },
        )

        trainer = Completion(
            pipeline,
            Fit(),
            monitor=Monitor(),
            device="cpu",
        )
        initial_losses = []
        final_losses = []
        for step_idx, batch in enumerate(loader):
            if step_idx == 0:
                initial_losses.append(float(trainer.step(batch)["loss"]))
            if step_idx >= 2:
                break
            final_losses.append(float(trainer.step(batch)["loss"]))

        assert len(initial_losses) >= 1
        assert len(final_losses) >= 1
        assert all(np.isfinite(v) for v in initial_losses + final_losses)

    def pipeline(self) -> None:
        seed_everything(1)
        rng = np.random.default_rng(1)
        n = 32
        features_np = {
            "visual": rng.normal(size=(n, 6)).astype(np.float32),
            "text": rng.normal(size=(n, 3)).astype(np.float32),
        }
        mask_np = np.ones((n, 2), dtype=np.float32)
        adj = build(n)

        config = Config()
        pipeline = Pipeline(config, dims={"visual": 6, "text": 3})
        pipeline.attach(features_np, mask_np, adj)
        pipeline.train()

        features = {k: torch.from_numpy(v) for k, v in features_np.items()}
        mask = torch.from_numpy(mask_np)
        index = torch.arange(n)
        out = pipeline(features, mask, adj, index=index, training=True)

        probs = out.routing
        assert probs.shape == (n, config.codebook.size)
        assert torch.all(probs >= 0)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(n), atol=1e-4)
        code_usage = probs.mean(dim=0)
        assert (code_usage > 0).any()
