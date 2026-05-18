#!/usr/bin/env python3
"""End-to-end demo of GRE-MC on synthetic data."""

import numpy as np
import torch

from rmr.data.graph_builder import build_item_graph, build_user_item_graph
from rmr.data.masking import apply_modality_mask
from rmr.evaluation.evaluator import Evaluator
from rmr.models.downstream import LightGCN
from rmr.models.gre_mc import GREMC


def generate_synthetic_data(n_users: int = 20, n_items: int = 50) -> tuple:
    """Generates a small synthetic dataset for demonstration.

    Args:
        n_users: Number of synthetic users.
        n_items: Number of synthetic items.

    Returns:
        Tuple of (user_item_graph, item_item_graph, features, mask).
    """
    uids = np.random.randint(0, n_users, size=200)
    iids = np.random.randint(0, n_items, size=200)
    ui = build_user_item_graph(uids, iids, n_users, n_items)
    ii = build_item_graph(ui)
    features = {
        "visual": np.random.randn(n_items, 16).astype(np.float32),
        "text": np.random.randn(n_items, 8).astype(np.float32),
    }
    mask = apply_modality_mask(n_items, 2, mask_ratio=0.4, seed=42)
    return ui, ii, features, mask


def main() -> None:
    """Runs the end-to-end demo."""
    print("Generating synthetic data...")
    ui, ii, features, mask = generate_synthetic_data()
    n_users, n_items = ui.shape

    print("Building GRE-MC model...")
    model = GREMC(
        input_dims={"visual": 16, "text": 8},
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
        codebook_size=10,
        top_p=2,
        tau=0.5,
        pe_dim=4,
    )
    model.register_retrieval_buffers(features, mask)
    feats_torch = {k: torch.from_numpy(v) for k, v in features.items()}
    mask_torch = torch.from_numpy(mask)
    x_hat, g = model(feats_torch, mask_torch, ii, training=True)
    print(f"  Reconstructed visual shape: {x_hat['visual'].shape}")
    print(f"  Routing weights shape: {g.shape}")

    print("Building LightGCN recommender...")
    recommender = LightGCN(n_users, n_items, embedding_dim=32, num_layers=2)
    scores = recommender(torch.arange(n_users), torch.arange(n_items), ui)
    print(f"  Score matrix shape: {scores.shape}")

    print("Evaluating...")
    labels = ui.sign().toarray()
    ev = Evaluator(ks=[10])
    res = ev.evaluate(scores.detach().numpy(), labels)
    for k, v in res.items():
        print(f"  {k}: {v:.4f}")

    print("Demo complete.")


if __name__ == "__main__":
    main()
