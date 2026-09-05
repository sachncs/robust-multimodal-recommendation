"""Benchmarks for morel forward pass latency at scale."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from morel.core.config import Config
from morel.pipeline import Pipeline


def _make_pipeline(items: int, dims: dict[str, int]) -> Pipeline:
    rng = np.random.default_rng(0)
    features = {k: rng.normal(size=(items, d)).astype(np.float32) for k, d in dims.items()}
    mask = np.ones((items, len(dims)), dtype=np.float32)
    adjacency = sp.csr_matrix(np.eye(items, dtype=np.float32))
    config = Config()
    pipeline = Pipeline(config, dims=dims)
    pipeline.register_buffers(features, mask, adjacency)
    return pipeline


def bench_forward_1k(benchmark) -> None:
    pipeline = _make_pipeline(1000, {"visual": 16, "text": 8})
    features = {k: torch.from_numpy(v) for k, v in pipeline._retrieval_features.items()}
    mask = torch.from_numpy(pipeline._retrieval_mask)
    index = torch.arange(1000)

    def run():
        pipeline(features, mask, pipeline._retrieval_adj, index=index, training=False)

    benchmark(run)


def bench_forward_10k(benchmark) -> None:
    pipeline = _make_pipeline(10000, {"visual": 16, "text": 8})
    features = {k: torch.from_numpy(v) for k, v in pipeline._retrieval_features.items()}
    mask = torch.from_numpy(pipeline._retrieval_mask)
    index = torch.arange(10000)

    def run():
        pipeline(features, mask, pipeline._retrieval_adj, index=index, training=False)

    benchmark(run)
