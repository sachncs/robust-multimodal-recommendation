"""Integration test: real Pipeline driven by Completion trainer."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

from morel.core.config import Config
from morel.pipeline import Pipeline
from morel.train.completion import Completion, CompletionConfig


class _Monitor:
    def log(self, *args, **kwargs):  # noqa: ANN001, D401
        return None


def _build_path_graph(n: int) -> sp.csr_matrix:
    rows, cols = [], []
    for i in range(n - 1):
        rows.extend([i, i + 1])
        cols.extend([i + 1, i])
    return sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n, n),
    )


def test_real_pipeline_completion_fit(tmp_path) -> None:
    rng = np.random.default_rng(0)
    n = 30
    dim_visual = 6
    dim_text = 4
    features_np = {
        "visual": rng.normal(size=(n, dim_visual)).astype(np.float32),
        "text": rng.normal(size=(n, dim_text)).astype(np.float32),
    }
    mask_np = np.ones((n, 2), dtype=np.float32)

    adj = _build_path_graph(n)

    config = Config()
    pipeline = Pipeline(config, dims={"visual": dim_visual, "text": dim_text})
    pipeline.attach_corpus(features_np, mask_np, adj)

    class _Ds(Dataset):
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
        _Ds(),
        batch_size=8,
        collate_fn=lambda batch: {
            "index": torch.from_numpy(
                np.stack([np.asarray(b["index"]) for b in batch])
            ),
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
        CompletionConfig(),
        monitor=_Monitor(),
        checkpoint_dir=tmp_path,
        device="cpu",
    )
    result = trainer.fit(loader, None, epochs=2, patience=4)
    assert "best" in result
    assert torch.isfinite(torch.tensor(result["best"])) or result["best"] == float("inf")


def test_real_pipeline_with_one_modality_missing(tmp_path) -> None:
    """Half the items have their text modality masked."""
    rng = np.random.default_rng(1)
    n = 24
    features_np = {
        "visual": rng.normal(size=(n, 4)).astype(np.float32),
        "text": rng.normal(size=(n, 2)).astype(np.float32),
    }
    mask_np = np.ones((n, 2), dtype=np.float32)
    mask_np[n // 2 :, 1] = 0.0
    adj = _build_path_graph(n)

    pipeline = Pipeline(Config(), dims={"visual": 4, "text": 2})
    pipeline.attach_corpus(features_np, mask_np, adj)

    features_t = {k: torch.from_numpy(v) for k, v in features_np.items()}
    mask_t = torch.from_numpy(mask_np)
    index = torch.arange(n)
    out = pipeline(features_t, mask_t, adj, index=index, training=False)
    assert out.completed["visual"].shape == (n, 4)
    assert out.completed["text"].shape == (n, 2)
    assert torch.isfinite(out.completed["text"]).all()


def test_real_pipeline_rejects_self_loops() -> None:
    """Pipeline raises GraphError when an adjacency with self-loops is passed."""
    from morel.core.errors import GraphError

    rng = np.random.default_rng(2)
    n = 8
    features_np = {
        "visual": rng.normal(size=(n, 4)).astype(np.float32),
        "text": rng.normal(size=(n, 2)).astype(np.float32),
    }
    mask_np = np.ones((n, 2), dtype=np.float32)
    bad = sp.csr_matrix(np.eye(n, dtype=np.float32))
    pipeline = Pipeline(Config(), dims={"visual": 4, "text": 2})
    pipeline.attach_corpus(features_np, mask_np, bad)

    features_t = {k: torch.from_numpy(v) for k, v in features_np.items()}
    mask_t = torch.from_numpy(mask_np)
    import pytest

    with pytest.raises(GraphError):
        pipeline(features_t, mask_t, bad, training=False)
