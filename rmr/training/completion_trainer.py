"""Trainer for the GRE-MC modality completion stage."""


import scipy.sparse as sp
import torch
import torch.nn as nn
from tqdm import tqdm


class CompletionTrainer:
    """Trainer for the modality completion stage.

    Encapsulates the training loop, loss computation, and optimization for
    the GRE-MC model.  The adjacency matrix is stored at initialization since
    it is constant across all batches.
    """

    def __init__(
        self,
        model: nn.Module,
        adjacency: sp.spmatrix,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        lambda_usage: float = 1.0,
        lambda_load: float = 1.0,
        device: str = "cpu",
    ) -> None:
        """Initializes the trainer.

        Args:
            model: The GRE-MC model to train.
            adjacency: Sparse item-item adjacency matrix.
            lr: Learning rate for Adam.
            weight_decay: L2 weight decay.
            lambda_usage: Weight for the codebook usage regularizer.
            lambda_load: Weight for the codebook load regularizer.
            device: Device to run training on.
        """
        self.model = model.to(device)
        self.adjacency = adjacency
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.lambda_usage = lambda_usage
        self.lambda_load = lambda_load

    def compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        mask: torch.Tensor,
        g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the total completion loss.

        The reconstruction loss is computed only on modalities that are masked
        (missing) according to ``mask``.  Usage and load regularizers are
        applied to the codebook routing weights.

        Args:
            predictions: Dict of predicted feature tensors.
            targets: Dict of ground-truth feature tensors.
            mask: Binary indicator matrix of shape (B, M).  A value of 0 means
                the modality is missing and should be reconstructed.
            g: Routing weight matrix of shape (B, codebook_size).

        Returns:
            Tuple of (total_loss, recon_loss, usage_loss, load_loss).
        """
        recon_loss = torch.tensor(0.0, device=mask.device)
        mods = list(predictions.keys())
        missing_count = 0.0
        for i, m in enumerate(mods):
            missing = (mask[:, i] == 0).float().unsqueeze(1)
            diff = predictions[m] - targets[m]
            recon_loss += (missing * (diff ** 2)).sum()
            missing_count += missing.sum()
        recon_loss = recon_loss / (missing_count + 1e-10)
        l_usage = self.model.codebook.usage_loss(g)
        l_load = self.model.codebook.load_loss(g)
        total = (
            recon_loss
            + self.lambda_usage * l_usage
            + self.lambda_load * l_load
        )
        return total, recon_loss, l_usage, l_load

    def train_epoch(self, dataloader) -> float:
        """Trains the model for one epoch.

        Args:
            dataloader: DataLoader yielding batches with keys ``index``,
                ``features``, and ``mask``.

        Returns:
            Average loss over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(dataloader, desc="Training"):
            index = batch["index"].to(self.device)
            features = {
                k: v.to(self.device) for k, v in batch["features"].items()
            }
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()
            predictions, g = self.model(
                features,
                mask,
                self.adjacency,
                index=index,
                training=True,
            )
            loss, recon_loss, l_usage, l_load = self.compute_loss(
                predictions, features, mask, g
            )
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(dataloader), 1)
