"""Completion trainer: drives the GRE-MC completion stage."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from morel.train.checkpoint import hash_config
from morel.train.loss import Reconstruction
from morel.train.monitor import Monitor
from morel.train.trainer import Trainer


@dataclass
class CompletionConfig:
    """Configuration for completion training."""

    lambda_usage: float = 1.0
    lambda_balance: float = 1.0
    grad_clip: float = 1.0

    def hash(self) -> str:
        import hashlib
        import json

        from dataclasses import asdict

        raw = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Completion(Trainer):
    """Trainer for modality completion."""

    def __init__(
        self,
        model,
        config,
        *,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        monitor: Monitor | None = None,
        checkpoint_dir=None,
    ) -> None:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        loss = Reconstruction()
        super().__init__(
            model=model,
            optimizer=optimizer,
            loss=loss,
            config=config,
            monitor=monitor,
            checkpoint_dir=checkpoint_dir,
            grad_clip=config.grad_clip if hasattr(config, "grad_clip") else 1.0,
        )
        self.completion_config = config

    def step(self, batch: dict) -> dict:
        """One optimisation step."""
        features = {k: v.to(self.device) for k, v in batch["features"].items()}
        mask = batch["mask"].to(self.device)
        index = batch.get("index")
        if index is not None:
            index = index.to(self.device)
        self.optimizer.zero_grad()
        predictions, probs = self.model(
            features, mask, batch["adjacency"], index=index, training=True
        )
        recon = self.loss.forward(predictions, features, mask)
        usage_term = self.model.codebook.usage(probs)
        balance_term = self.model.codebook.balance(probs)
        total = recon + self.completion_config.lambda_usage * usage_term + self.completion_config.lambda_balance * balance_term
        total.backward()
        self.clip(list(self.model.parameters()))
        self.optimizer.step()
        return {"loss": float(total.item()), "recon": float(recon.item())}

    def validate(self, loader: DataLoader) -> float:
        """Return the mean reconstruction loss on the validation set."""
        self.model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                features = {k: v.to(self.device) for k, v in batch["features"].items()}
                mask = batch["mask"].to(self.device)
                index = batch.get("index")
                if index is not None:
                    index = index.to(self.device)
                predictions, _ = self.model(
                    features, mask, batch["adjacency"], index=index, training=False
                )
                recon = self.loss.forward(predictions, features, mask)
                total += float(recon.item())
                count += 1
        return total / max(count, 1)


__all__ = ["Completion", "CompletionConfig"]
