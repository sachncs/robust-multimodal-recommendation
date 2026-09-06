"""Completion trainer: drives the GRE-MC completion stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from morel.codebook.codebook import balance, usage
from morel.train.loss import Loss, Reconstruction
from morel.train.monitor import Monitor
from morel.train.trainer import Trainer


@dataclass
class FitConfig:
    """Configuration for completion training."""

    lambda_usage: float = 1.0
    lambda_balance: float = 1.0
    grad_clip: float = 1.0


class Completion(Trainer):
    """Trainer for modality completion."""

    def __init__(
        self,
        model: nn.Module,
        config: FitConfig,
        *,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        monitor: Monitor | None = None,
        checkpoint_dir: Path | str | None = None,
        device: str | torch.device | None = None,
        amp: bool = False,
    ) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        reconstruction: Loss = Reconstruction()
        super().__init__(
            model=model,
            optimizer=optimizer,
            loss=reconstruction,
            config=config,
            monitor=monitor,
            checkpoint_dir=checkpoint_dir,
            grad_clip=config.grad_clip if hasattr(config, "grad_clip") else 1.0,
            device=device,
            amp=amp,
        )
        self.completion_config = config
        self.reconstruction = reconstruction

    def step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """One optimisation step."""
        features = {k: v.to(self.device) for k, v in batch["features"].items()}
        mask = batch["mask"].to(self.device)
        index = batch.get("index")
        if index is not None:
            index = index.to(self.device)
        self.optimizer.zero_grad()
        with self.autocast():
            output = self.model(features, mask, batch["adjacency"], index=index, training=True)
            predictions = output.completed
            probs = output.routing
            recon = self.reconstruction.forward(predictions, features, mask)
            usage_term = usage(probs)
            balance_term = balance(probs)
            total = (
                recon
                + self.completion_config.lambda_usage * usage_term
                + self.completion_config.lambda_balance * balance_term
            )
        if self.scaler is not None:
            self.scaler.scale(total).backward()  # type: ignore[no-untyped-call]  # torch stubs leave this untyped
            self.scaler.unscale_(self.optimizer)
            self.clip(list(self.model.parameters()))
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total.backward()  # type: ignore[no-untyped-call]  # torch stubs leave this untyped
            self.clip(list(self.model.parameters()))
            self.optimizer.step()
        return {"loss": float(total.item()), "recon": float(recon.item())}

    def validate(self, loader: DataLoader[Any]) -> float:
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
                output = self.model(features, mask, batch["adjacency"], index=index, training=False)
                recon = self.reconstruction.forward(output.completed, features, mask)
                total += float(recon.item())
                count += 1
        return total / max(count, 1)


__all__ = ["Completion", "FitConfig"]
