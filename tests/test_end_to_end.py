import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader

from rmr.data.dataset import GREMCGraphDataset
from rmr.data.graph_builder import build_item_graph, build_user_item_graph
from rmr.data.masking import apply_modality_mask
from rmr.evaluation.metrics import recall_at_k
from rmr.models.downstream import LightGCN
from rmr.models.gre_mc import GREMC
from rmr.training.completion_trainer import CompletionTrainer


def test_end_to_end_shapes():
    n_users = 5
    n_items = 10
    # Synthetic interactions
    uids = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    iids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
    ui = build_user_item_graph(uids, iids, n_users, n_items)
    ii = build_item_graph(ui)
    # Features
    features = {
        "visual": np.random.randn(n_items, 10).astype(np.float32),
        "text": np.random.randn(n_items, 8).astype(np.float32),
    }
    mask = apply_modality_mask(n_items, 2, mask_ratio=0.4, seed=42)
    # GRE-MC model
    model = GREMC(
        input_dims={"visual": 10, "text": 8},
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        codebook_size=5,
        top_p=2,
        tau=0.5,
        pe_dim=4,
    )
    feats_torch = {k: torch.from_numpy(v) for k, v in features.items()}
    mask_torch = torch.from_numpy(mask)
    x_hat, g = model(feats_torch, mask_torch, ii, training=True)
    assert x_hat["visual"].shape == (n_items, 10)
    assert x_hat["text"].shape == (n_items, 8)
    # LightGCN
    recommender = LightGCN(num_users=n_users, num_items=n_items, embedding_dim=16, num_layers=2)
    scores = recommender(torch.arange(n_users), torch.arange(n_items), ui)
    assert scores.shape == (n_users, n_items)


def test_train_epoch_updates_gremc_weights():
    """Integration test: train_epoch should change GREMC weights."""
    n_items = 6
    adj = sp.csr_matrix(np.eye(n_items), dtype=np.float32)
    features = {
        "visual": np.random.randn(n_items, 4).astype(np.float32),
        "text": np.random.randn(n_items, 2).astype(np.float32),
    }
    mask = np.zeros((n_items, 2), dtype=np.float32)
    dataset = GREMCGraphDataset(features, mask, adj)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)

    model = GREMC(
        input_dims={"visual": 4, "text": 2},
        hidden_dim=8,
        num_layers=1,
        num_heads=1,
        codebook_size=4,
        top_p=2,
        tau=0.5,
        pe_dim=2,
        dropout=0.0,
    )
    trainer = CompletionTrainer(model, adjacency=adj, lr=0.01)

    initial_weights = {
        name: param.clone()
        for name, param in model.named_parameters()
    }
    loss = trainer.train_epoch(dataloader)
    assert loss > 0.0
    updated = sum(
        1 for name, param in model.named_parameters()
        if not torch.equal(initial_weights[name], param)
    )
    assert updated > 0


def test_gremc_output_wired_into_lightgcn():
    """Integration test: use GREMC routing weights as LightGCN item embeddings."""
    n_users = 4
    n_items = 8
    uids = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    iids = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    ui = build_user_item_graph(uids, iids, n_users, n_items)
    ii = build_item_graph(ui)

    codebook_size = 8
    features = {
        "visual": np.random.randn(n_items, 4).astype(np.float32),
    }
    mask = np.ones((n_items, 1), dtype=np.float32)
    model = GREMC(
        input_dims={"visual": 4},
        hidden_dim=codebook_size,
        num_layers=1,
        num_heads=1,
        codebook_size=codebook_size,
        top_p=2,
        tau=0.5,
        pe_dim=2,
        dropout=0.0,
    )
    feats_torch = {k: torch.from_numpy(v) for k, v in features.items()}
    mask_torch = torch.from_numpy(mask)
    _, g = model(feats_torch, mask_torch, ii, training=False)

    # Wire routing weights into LightGCN item embeddings.
    recommender = LightGCN(
        num_users=n_users,
        num_items=n_items,
        embedding_dim=codebook_size,
        num_layers=1,
    )
    with torch.no_grad():
        recommender.item_emb.weight.copy_(g)

    scores = recommender(torch.arange(n_users), torch.arange(n_items), ui)
    assert scores.shape == (n_users, n_items)
    # Scores should be finite and the wiring should not produce NaNs.
    assert torch.isfinite(scores).all()
    labels = ui.sign().toarray().astype(np.int32)
    r = recall_at_k(scores.detach().numpy(), labels, k=2)
    assert 0.0 <= r <= 1.0
