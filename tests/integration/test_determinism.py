"""End-to-end determinism contract.

Regression coverage for the case that motivated the seeding work: running the
same configured pipeline twice in one process used to produce different
metrics, because model parameters were drawn from the ambient global RNG and
dropout stayed active during inference.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from morel.core.config import Config
from morel.data.build import bipartite, item_cooccurrence
from morel.data.mask import bernoulli
from morel.eval import ndcg_at_k, recall_at_k
from morel.pipeline import Pipeline
from morel.recommend import Light


def synthetic_corpus() -> tuple[dict[str, np.ndarray], np.ndarray, sp.csr_matrix, sp.csr_matrix]:
    """Build the small synthetic user-item problem used by the demo."""
    rng = np.random.default_rng(0)
    users, items = 20, 50
    ui = bipartite(rng.integers(0, users, size=200), rng.integers(0, items, size=200), users, items)
    features = {
        "visual": rng.normal(size=(items, 16)).astype(np.float32),
        "text": rng.normal(size=(items, 8)).astype(np.float32),
    }
    mask = bernoulli(items, 2, 0.4, seed=42).to_numpy()
    return features, mask, item_cooccurrence(ui), ui


def run_once(config: Config) -> tuple[dict[str, torch.Tensor], torch.Tensor, float, float]:
    """Run the demo flow end to end and return its outputs and metrics."""
    features, mask, adjacency, ui = synthetic_corpus()
    items = mask.shape[0]
    users = ui.shape[0]

    pipeline = Pipeline(config, dims={"visual": 16, "text": 8})
    pipeline.attach(features, mask, adjacency)
    output = pipeline(
        {name: torch.from_numpy(value) for name, value in features.items()},
        torch.from_numpy(mask),
        adjacency,
        index=torch.arange(items),
        training=False,
    )

    recommender = Light(users=users, items=items, embed=32, layers=2, seed=config.seed)
    scores = recommender(torch.arange(users), torch.arange(items), ui).detach().numpy()
    labels = ui.sign().toarray()
    return (
        output.completed,
        output.routing,
        float(recall_at_k(scores, labels, k=10)),
        float(ndcg_at_k(scores, labels, k=10)),
    )


def test_same_config_yields_identical_metrics() -> None:
    config = Config()
    torch.manual_seed(1)
    _, _, recall_a, ndcg_a = run_once(config)
    torch.manual_seed(9999)
    _, _, recall_b, ndcg_b = run_once(config)

    assert recall_a == recall_b
    assert ndcg_a == ndcg_b


def test_same_config_yields_identical_tensors() -> None:
    config = Config()
    torch.manual_seed(1)
    completed_a, routing_a, _, _ = run_once(config)
    torch.manual_seed(9999)
    completed_b, routing_b, _, _ = run_once(config)

    assert torch.equal(routing_a, routing_b)
    assert completed_a.keys() == completed_b.keys()
    for name in completed_a:
        assert torch.equal(completed_a[name], completed_b[name]), f"modality {name} differs"


def test_different_seed_changes_the_model() -> None:
    """Determinism must come from the seed, not from the model being constant."""
    _, routing_a, _, _ = run_once(Config())
    _, routing_b, _, _ = run_once(Config(seed=7))

    assert not torch.equal(routing_a, routing_b)
