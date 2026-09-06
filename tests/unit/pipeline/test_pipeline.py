"""Tests for morel.pipeline."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from morel.core.config import Config
from morel.pipeline import Pipeline


def test_pipeline_end_to_end() -> None:
    config = Config.defaults()
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
    pipeline.attach_corpus(features_np, mask_np, adjacency)
    index = torch.tensor([0, 1, 2])
    features = {k: torch.from_numpy(v[index.numpy()]) for k, v in features_np.items()}
    mask = torch.from_numpy(mask_np[index.numpy()])
    out = pipeline(features, mask, adjacency, index=index, training=False)
    assert out.completed["visual"].shape == (3, 4)
    assert out.subgraph_indices is not None
    assert out.subgraph_mask is not None


def test_pipeline_weights_are_determined_by_config_seed() -> None:
    """Two pipelines from one config must match despite differing ambient RNG."""
    config = Config()
    torch.manual_seed(1)
    first = Pipeline(config, dims={"visual": 4, "text": 2})
    torch.manual_seed(9999)
    second = Pipeline(config, dims={"visual": 4, "text": 2})

    left, right = first.state_dict(), second.state_dict()
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), f"parameter {key} differs"


def test_pipeline_construction_preserves_caller_rng_state() -> None:
    config = Config()
    torch.manual_seed(7)
    expected = torch.randn(4)

    torch.manual_seed(7)
    Pipeline(config, dims={"visual": 4, "text": 2})
    actual = torch.randn(4)

    assert torch.equal(expected, actual)


def test_pipeline_forward_is_reproducible_across_instances() -> None:
    config = Config()
    features = {"visual": torch.randn(3, 4), "text": torch.randn(3, 2)}
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    adjacency = sp.csr_matrix(np.zeros((3, 3), dtype=np.float32))

    first = Pipeline(config, dims={"visual": 4, "text": 2})(
        features, mask, adjacency, training=False
    )
    second = Pipeline(config, dims={"visual": 4, "text": 2})(
        features, mask, adjacency, training=False
    )

    assert torch.equal(first.routing, second.routing)
    for name in first.completed:
        assert torch.equal(first.completed[name], second.completed[name])


def test_inference_pass_disables_dropout() -> None:
    """Regression: training=False left nn.Dropout active, so inference was noisy."""
    config = Config()
    assert config.encode.dropout > 0, "test needs a nonzero dropout to be meaningful"
    pipeline = Pipeline(config, dims={"visual": 4, "text": 2})
    features = {"visual": torch.randn(3, 4), "text": torch.randn(3, 2)}
    mask = torch.ones(3, 2)
    adjacency = sp.csr_matrix(np.zeros((3, 3), dtype=np.float32))

    first = pipeline(features, mask, adjacency, training=False)
    second = pipeline(features, mask, adjacency, training=False)

    assert torch.equal(first.routing, second.routing)
    for name in first.completed:
        assert torch.equal(first.completed[name], second.completed[name])


def test_forward_restores_module_mode() -> None:
    config = Config()
    pipeline = Pipeline(config, dims={"visual": 4, "text": 2})
    features = {"visual": torch.randn(3, 4), "text": torch.randn(3, 2)}
    mask = torch.ones(3, 2)
    adjacency = sp.csr_matrix(np.zeros((3, 3), dtype=np.float32))

    assert pipeline.training is True
    pipeline(features, mask, adjacency, training=False)
    assert pipeline.training is True, "an inference pass must not leave the module in eval mode"

    pipeline.eval()
    pipeline(features, mask, adjacency, training=True)
    assert pipeline.training is False, "a training pass must not leave the module in train mode"


def test_attach_recommender_primes_adjacency_cache() -> None:
    """Regression: attach_recommender used to raise AttributeError."""
    config = Config()
    pipeline = Pipeline(config, dims={"visual": 4, "text": 2})
    ui = sp.csr_matrix(np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float32))

    pipeline.attach_recommender(ui)

    assert pipeline.recommender is not None
    assert pipeline.recommender.users == 2
    assert pipeline.recommender.items == 3
    assert pipeline.recommender.adj_cache is not None
    scores = pipeline.recommender(torch.arange(2), torch.arange(3))
    assert scores.shape == (2, 3)


def test_attach_recommender_is_seeded_from_config() -> None:
    config = Config()
    ui = sp.csr_matrix(np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float32))

    first = Pipeline(config, dims={"visual": 4, "text": 2})
    torch.manual_seed(1)
    first.attach_recommender(ui)
    second = Pipeline(config, dims={"visual": 4, "text": 2})
    torch.manual_seed(4321)
    second.attach_recommender(ui)

    assert first.recommender is not None
    assert second.recommender is not None
    assert torch.equal(first.recommender.user_emb.weight, second.recommender.user_emb.weight)
