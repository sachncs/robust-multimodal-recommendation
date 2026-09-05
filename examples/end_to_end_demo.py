"""End-to-end demo of morel on synthetic data.

Run with::

    python examples/end_to_end_demo.py

This script:
1. Generates a small synthetic user-item graph and modality features.
2. Builds the GRE-MC pipeline.
3. Runs the completion stage forward.
4. Evaluates the downstream LightGCN recommender.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from morel.core.config import Config
from morel.data.build import bipartite, item_cooccurrence
from morel.data.mask import bernoulli
from morel.eval import recall_at_k, ndcg_at_k
from morel.pipeline import Pipeline
from morel.recommend import Light


def main() -> None:
    """Run the end-to-end demo."""
    rng = np.random.default_rng(0)
    users, items = 20, 50
    uids = rng.integers(0, users, size=200)
    iids = rng.integers(0, items, size=200)
    ui = bipartite(uids, iids, users, items)
    adjacency = item_cooccurrence(ui)

    features = {
        "visual": rng.normal(size=(items, 16)).astype(np.float32),
        "text": rng.normal(size=(items, 8)).astype(np.float32),
    }
    mask = bernoulli(items, 2, 0.4, seed=42).to_numpy()

    config = Config()
    pipeline = Pipeline(config, dims={"visual": 16, "text": 8})
    pipeline.register_buffers(features, mask, adjacency)

    features_t = {k: torch.from_numpy(v) for k, v in features.items()}
    mask_t = torch.from_numpy(mask)
    index = torch.arange(items)
    output = pipeline(features_t, mask_t, adjacency, index=index, training=False)
    print(f"Reconstructed visual shape: {tuple(output.completed['visual'].shape)}")
    print(f"Routing weights shape: {tuple(output.routing.shape)}")

    recommender = Light(users=users, items=items, embed=32, layers=2)
    scores = recommender(torch.arange(users), torch.arange(items), ui)
    print(f"Score matrix shape: {tuple(scores.shape)}")

    labels = ui.sign().toarray()
    arr = scores.detach().numpy()
    print(f"  recall@10: {recall_at_k(arr, labels, k=10):.4f}")
    print(f"  ndcg@10: {ndcg_at_k(arr, labels, k=10):.4f}")


if __name__ == "__main__":
    main()
