"""Forward-pass latency benchmarks for morel at item scales 1k, 10k, 100k."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from morel.core.config import Config
from morel.pipeline import Pipeline


def make_pipeline(items: int, dims: dict[str, int]) -> Pipeline:
    """Build a corpus-attached :class:`Pipeline` over a ring item graph.

    The ring is connected and self-loop-free, which is required by
    :meth:`Pipeline.__call__`. Dimensions match the production benchmark.

    Args:
        items: Number of items in the corpus.
        dims: Mapping from modality name to its feature dimension.

    Returns:
        A configured :class:`Pipeline` with retrieval buffers attached.
    """
    rng = np.random.default_rng(0)
    features = {k: rng.normal(size=(items, d)).astype(np.float32) for k, d in dims.items()}
    mask = np.ones((items, len(dims)), dtype=np.float32)
    offsets = np.arange(items)
    rows = np.concatenate([offsets, (offsets + 1) % items])
    cols = np.concatenate([(offsets + 1) % items, offsets])
    adjacency = sp.csr_matrix(
        (np.ones(rows.size, dtype=np.float32), (rows, cols)),
        shape=(items, items),
    )
    pipeline = Pipeline(Config(), dims=dims)
    pipeline.attach(features, mask, adjacency)
    return pipeline


def bench_1k(benchmark) -> None:
    """Measure forward latency at one thousand items."""
    pipeline = make_pipeline(1000, {"visual": 16, "text": 8})
    features = {k: torch.from_numpy(v) for k, v in pipeline.retrieval_features.items()}
    mask = torch.from_numpy(pipeline.retrieval_mask)
    index = torch.arange(1000)

    def run() -> None:
        pipeline(features, mask, pipeline.retrieval_adj, index=index, training=False)

    benchmark(run)


def bench_10k(benchmark) -> None:
    """Measure forward latency at ten thousand items."""
    pipeline = make_pipeline(10000, {"visual": 16, "text": 8})
    features = {k: torch.from_numpy(v) for k, v in pipeline.retrieval_features.items()}
    mask = torch.from_numpy(pipeline.retrieval_mask)
    index = torch.arange(10000)

    def run() -> None:
        pipeline(features, mask, pipeline.retrieval_adj, index=index, training=False)

    benchmark(run)


def bench_100k(benchmark) -> None:
    """Measure forward latency at one hundred thousand items.

    This is a smoke benchmark — its primary purpose is to prove the path
    scales, not to enforce tight SLOs.
    """
    pipeline = make_pipeline(100000, {"visual": 16, "text": 8})
    features = {k: torch.from_numpy(v) for k, v in pipeline.retrieval_features.items()}
    mask = torch.from_numpy(pipeline.retrieval_mask)
    index = torch.arange(100000)

    def run() -> None:
        pipeline(features, mask, pipeline.retrieval_adj, index=index, training=False)

    benchmark(run)
