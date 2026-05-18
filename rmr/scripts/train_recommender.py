"""Training script for the downstream LightGCN recommender."""

import argparse
import os

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

from rmr.models.downstream import LightGCN


def bpr_loss(
    pos_scores: torch.Tensor, neg_scores: torch.Tensor
) -> torch.Tensor:
    """Computes Bayesian Personalized Ranking loss.

    Args:
        pos_scores: Scores for positive items, shape (B,).
        neg_scores: Scores for negative items, shape (B,).

    Returns:
        Scalar BPR loss.
    """
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()


def sample_negatives(
    n_users: int, n_items: int, interactions: sp.csr_matrix, num_negatives: int
) -> np.ndarray:
    """Samples negative items for each user.

    Args:
        n_users: Number of users.
        n_items: Number of items.
        interactions: User-item interaction matrix.
        num_negatives: Number of negatives to sample per user.

    Returns:
        Array of negative item IDs of shape (n_users, num_negatives).
    """
    negatives = np.zeros((n_users, num_negatives), dtype=np.int64)
    for u in range(n_users):
        pos_items = set(
            interactions.indices[
                interactions.indptr[u]:interactions.indptr[u + 1]
            ]
        )
        neg_candidates = list(set(range(n_items)) - pos_items)
        if len(neg_candidates) < num_negatives:
            neg_candidates = list(range(n_items))
        negatives[u] = np.random.choice(
            neg_candidates, size=num_negatives, replace=False
        )
    return negatives


def main() -> None:
    """Parses arguments and runs the LightGCN BPR training loop."""
    parser = argparse.ArgumentParser(
        description="Train the LightGCN downstream recommender."
    )
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-negatives", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    ui = sp.load_npz(os.path.join(args.data_dir, "user_item_graph.npz"))
    n_users, n_items = ui.shape
    model = LightGCN(
        n_users,
        n_items,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
    )
    model = model.to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Prepare positive item arrays for each user.
    user_pos_items = []
    for u in range(n_users):
        items = ui.indices[ui.indptr[u] : ui.indptr[u + 1]]
        user_pos_items.append(items)

    for epoch in range(args.epochs):
        model.train()
        negatives = sample_negatives(
            n_users, n_items, ui, args.num_negatives
        )
        epoch_loss = 0.0
        num_batches = 0

        # Shuffle users and batch them.
        user_indices = np.random.permutation(n_users)
        for start in tqdm(
            range(0, n_users, args.batch_size), desc=f"Epoch {epoch + 1}"
        ):
            batch_users = user_indices[start : start + args.batch_size]
            batch_pos = []
            batch_neg = []
            for u in batch_users:
                pos = np.random.choice(user_pos_items[u])
                batch_pos.append(pos)
                batch_neg.append(negatives[u, 0])

            users_t = torch.from_numpy(batch_users).to(args.device)
            pos_t = torch.tensor(batch_pos, device=args.device)
            neg_t = torch.tensor(batch_neg, device=args.device)

            scores = model(
                users_t, torch.arange(n_items).to(args.device), ui
            )
            pos_scores = scores[torch.arange(len(users_t)), pos_t]
            neg_scores = scores[torch.arange(len(users_t)), neg_t]

            loss = bpr_loss(pos_scores, neg_scores)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.4f}")
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(
            model.state_dict(),
            os.path.join(args.checkpoint_dir, "recommender_best.pt"),
        )


if __name__ == "__main__":
    main()
