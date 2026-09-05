"""Tests for morel.pipeline."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from morel.core.config import Config
from morel.pipeline import Pipeline


def test_pipeline_end_to_end() -> None:
    config = Config(encode=Config.__dataclass_fields__["encode"].default_factory())  # type: ignore[misc]
    config = Config()
    pipeline = Pipeline(config, dims={"visual": 4, "text": 2})
    features = {
        "visual": torch.randn(3, 4),
        "text": torch.randn(3, 2),
    }
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    adjacency = sp.csr_matrix(np.zeros((3, 3), dtype=np.float32))
    out = pipeline(features, mask, adjacency, training=False)
    assert out.completed["visual"].shape == (3, 4)
    assert out.completed["text"].shape == (3, 2)
    assert out.routing.shape == (3, config.codebook.size)


def test_pipeline_with_retrieval() -> None:
    config = Config()
    # Use a larger graph so PE has enough dimensions.
    n = 32
    rng = np.random.default_rng(0)
    features_np = {
        "visual": rng.normal(size=(n, 4)).astype(np.float32),
        "text": rng.normal(size=(n, 2)).astype(np.float32),
    }
    mask_np = np.ones((n, 2), dtype=np.float32)
    arr = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        arr[i, i + 1] = 1
        arr[i + 1, i] = 1
    adjacency = sp.csr_matrix(arr)
    pipeline = Pipeline(config, dims={"visual": 4, "text": 2})
    pipeline.register_buffers(features_np, mask_np, adjacency)
    index = torch.tensor([0, 1, 2])
    features = {k: torch.from_numpy(v[index.numpy()]) for k, v in features_np.items()}
    mask = torch.from_numpy(mask_np[index.numpy()])
    out = pipeline(features, mask, adjacency, index=index, training=False)
    assert out.completed["visual"].shape == (3, 4)
    assert out.subgraph_indices is not None
    assert out.subgraph_mask is not None
