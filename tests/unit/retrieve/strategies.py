"""The retrieval strategy must be selectable from configuration.

``config.retrieve.kind`` defaulted to "mage" and was never read: the pipeline
always ran anchor retrieval followed by MAGE expansion. Asking for a different
strategy, including the no-retrieval ablation, silently produced the full
method.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from morel.core.config import Config
from morel.core.errors import ConfigError
from morel.pipeline import Pipeline
from morel.retrieve import KIND as STRATEGIES, retrieve, retrieve_batch


def corpus() -> tuple[dict[str, np.ndarray], np.ndarray, sp.csr_matrix]:
    """Return a small connected item graph with per-item features."""
    nodes = 32
    rng = np.random.default_rng(0)
    features = {
        "visual": rng.normal(size=(nodes, 4)).astype(np.float32),
        "text": rng.normal(size=(nodes, 2)).astype(np.float32),
    }
    mask = np.ones((nodes, 2), dtype=np.float32)
    arr = np.zeros((nodes, nodes), dtype=np.float32)
    for i in range(nodes - 1):
        arr[i, i + 1] = arr[i + 1, i] = 1.0
    return features, mask, sp.csr_matrix(arr)


class Checker:
    """Aggregated test methods for this module."""

    def registered() -> None:
        assert set(STRATEGIES) >= {"mage", "acs", "anchor", "bfs"}

    def query(kind: str) -> None:
        features, mask, adjacency = corpus()
        subgraph = retrieve(5, features, mask, adjacency, anchors=4, iters=2, kind=kind)
        assert 5 in subgraph, "the query item must always be part of its own subgraph"
        assert subgraph <= set(range(adjacency.shape[0]))

    def differ() -> None:
        """Selection would be meaningless if every kind returned the same thing."""
        features, mask, adjacency = corpus()
        results = {
            kind: frozenset(retrieve(5, features, mask, adjacency, anchors=4, iters=2, kind=kind))
            for kind in ("mage", "acs", "anchor", "bfs")
        }
        assert len(set(results.values())) > 1, f"all strategies agreed: {results}"

    def features() -> None:
        """The no-retrieval ablation must depend on the graph only."""
        features, mask, adjacency = corpus()
        baseline = retrieve(5, features, mask, adjacency, anchors=4, iters=2, kind="bfs")

        rng = np.random.default_rng(99)
        scrambled = {
            name: rng.normal(size=arr.shape).astype(np.float32) for name, arr in features.items()
        }
        other = retrieve(5, scrambled, mask, adjacency, anchors=4, iters=2, kind="bfs")

        assert baseline == other

    def budget() -> None:
        features, mask, adjacency = corpus()
        near = retrieve(10, features, mask, adjacency, anchors=4, iters=1, kind="bfs")
        far = retrieve(10, features, mask, adjacency, anchors=4, iters=3, kind="bfs")
        assert near < far, "a larger hop budget must reach at least as far"

    def rejected() -> None:
        features, mask, adjacency = corpus()
        with pytest.raises(ConfigError, match="unknown retrieval strategy 'nope'; available: "):
            retrieve(0, features, mask, adjacency, kind="nope")

    def strategy() -> None:
        features, mask, adjacency = corpus()
        queries = list(range(6))
        batched = retrieve_batch(queries, features, mask, adjacency, anchors=4, iters=2, kind="bfs")
        for row, query in enumerate(queries):
            expected = retrieve(query, features, mask, adjacency, anchors=4, iters=2, kind="bfs")
            got = set(batched.nodes[row, : batched.sizes[row]].tolist())
            assert got == expected

    def kind(kind: str) -> None:
        features, mask, adjacency = corpus()
        config = Config.from_dict(
            {
                "encode": {"hidden": 16, "pe": 4, "layers": 1, "heads": 2},
                "codebook": {"size": 16},
                "retrieve": {"kind": kind, "anchors": 4, "iters": 2},
            }
        )
        pipeline = Pipeline(config, dims={"visual": 4, "text": 2})
        pipeline.attach(features, mask, adjacency)
        out = pipeline(
            {name: torch.from_numpy(value[:6]) for name, value in features.items()},
            torch.from_numpy(mask[:6]),
            adjacency,
            index=torch.arange(6),
            training=False,
        )
        assert out.completed["visual"].shape == (6, 4)
        assert out.subgraph_indices is not None
