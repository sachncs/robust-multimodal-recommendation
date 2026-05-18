"""Tests for the modality completion trainer."""

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader

from rmr.data.dataset import GREMCGraphDataset
from rmr.models.gre_mc import GREMC
from rmr.training.completion_trainer import CompletionTrainer


def test_trainer_initialization():
    adj = sp.csr_matrix(np.eye(5), dtype=np.float32)
    model = torch.nn.Linear(4, 2)
    trainer = CompletionTrainer(model, adjacency=adj, lr=0.001)
    assert trainer.optimizer is not None
    assert trainer.adjacency is adj


def test_compute_loss_all_present():
    """When all modalities are present, recon loss should be zero."""
    adj = sp.csr_matrix(np.eye(3), dtype=np.float32)
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
    trainer = CompletionTrainer(model, adjacency=adj)
    features = {
        "visual": torch.randn(3, 4),
        "text": torch.randn(3, 2),
    }
    mask = torch.ones(3, 2)
    predictions = features  # perfect predictions
    g = torch.ones(3, 4) / 4  # uniform routing
    total, recon, usage, load = trainer.compute_loss(
        predictions, features, mask, g
    )
    assert recon.item() == 0.0
    assert total.item() >= 0.0


def test_compute_loss_all_missing():
    """When all modalities are missing, recon loss should be MSE over missing."""
    adj = sp.csr_matrix(np.eye(3), dtype=np.float32)
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
    trainer = CompletionTrainer(model, adjacency=adj)
    features = {
        "visual": torch.ones(3, 4),
        "text": torch.ones(3, 2),
    }
    predictions = {
        "visual": torch.zeros(3, 4),
        "text": torch.zeros(3, 2),
    }
    mask = torch.zeros(3, 2)
    g = torch.ones(3, 4) / 4
    total, recon, usage, load = trainer.compute_loss(
        predictions, features, mask, g
    )
    # MSE summed over all missing elements, divided by missing count.
    # 3 items * (4 + 2) dims = 18 total error; missing_count = 3*2 = 6.
    expected_recon = 18.0 / 6.0
    assert abs(recon.item() - expected_recon) < 1e-4


def test_compute_loss_mixed_mask():
    """Reconstruction loss applies only to masked (missing) modalities."""
    adj = sp.csr_matrix(np.eye(2), dtype=np.float32)
    model = GREMC(
        input_dims={"visual": 2, "text": 2},
        hidden_dim=4,
        num_layers=1,
        num_heads=1,
        codebook_size=2,
        top_p=1,
        tau=0.5,
        pe_dim=1,
        dropout=0.0,
    )
    trainer = CompletionTrainer(model, adjacency=adj)
    features = {
        "visual": torch.ones(2, 2),
        "text": torch.ones(2, 2),
    }
    predictions = {
        "visual": torch.zeros(2, 2),
        "text": torch.zeros(2, 2),
    }
    # First item: visual missing, text present.
    # Second item: both present.
    mask = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    g = torch.ones(2, 2) / 2
    total, recon, usage, load = trainer.compute_loss(
        predictions, features, mask, g
    )
    # Only the first item's visual modality is missing -> 2 errors.
    # Normalized by missing count (1 missing modality-item pair).
    expected_recon = 2.0 / 1.0
    assert abs(recon.item() - expected_recon) < 1e-4


def test_train_epoch_updates_weights():
    """train_epoch should change model weights when data is imperfect."""
    n_items = 4
    adj = sp.csr_matrix(np.eye(n_items), dtype=np.float32)
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
    features = {
        "visual": np.random.randn(n_items, 4).astype(np.float32),
        "text": np.random.randn(n_items, 2).astype(np.float32),
    }
    mask = np.zeros((n_items, 2), dtype=np.float32)
    dataset = GREMCGraphDataset(features, mask, adj)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Snapshot initial weights
    initial_weights = {
        name: param.clone()
        for name, param in model.named_parameters()
    }

    loss = trainer.train_epoch(dataloader)

    assert loss > 0.0
    changed = 0
    for name, param in model.named_parameters():
        if not torch.equal(initial_weights[name], param):
            changed += 1
    assert changed > 0
