"""Training script for the GRE-MC modality completion stage."""

import argparse
import os

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader

from rmr.data.dataset import GREMCGraphDataset
from rmr.models.gre_mc import GREMC
from rmr.training.completion_trainer import CompletionTrainer


def main() -> None:
    """Parses arguments and runs the completion training loop."""
    parser = argparse.ArgumentParser(
        description="Train the GRE-MC modality completion model."
    )
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=100)
    parser.add_argument("--top-p", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--pe-dim", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-usage", type=float, default=1.0)
    parser.add_argument("--lambda-load", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    print("Loading data...")
    features = {
        k: np.load(os.path.join(args.data_dir, f"{k}_features.npy"))
        for k in ["visual", "text"]
    }
    mask = np.load(os.path.join(args.data_dir, "mask.npy"))
    adj = sp.load_npz(os.path.join(args.data_dir, "item_graph.npz"))

    dataset = GREMCGraphDataset(features, mask, adj)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )

    model = GREMC(
        input_dims={k: v.shape[1] for k, v in features.items()},
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        codebook_size=args.codebook_size,
        top_p=args.top_p,
        tau=args.tau,
        pe_dim=args.pe_dim,
        dropout=args.dropout,
    )
    model.register_retrieval_buffers(features, mask)
    trainer = CompletionTrainer(
        model,
        adjacency=adj,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_usage=args.lambda_usage,
        lambda_load=args.lambda_load,
        device=args.device,
    )

    best_loss = float("inf")
    patience_counter = 0
    for epoch in range(args.epochs):
        loss = trainer.train_epoch(dataloader)
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss:.4f}")
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(args.checkpoint_dir, "completion_best.pt"),
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping.")
                break


if __name__ == "__main__":
    main()
