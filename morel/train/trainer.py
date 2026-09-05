"""Base trainer: composes model, optimizer, scheduler, loss, monitor, checkpoint."""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from morel.core.device import device as resolve_device
from morel.train.checkpoint import State, hash_config
from morel.train.monitor import Monitor


class Trainer(ABC):
    """Abstract training loop."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: object,
        config: object,
        *,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        monitor: Monitor | None = None,
        checkpoint_dir: Path | str | None = None,
        grad_clip: float | None = None,
        amp: bool = False,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.config = config
        self.grad_clip = grad_clip
        self.amp = amp
        self.device = resolve_device(device)
        self.model.to(self.device)
        self.monitor = monitor or Monitor(Path("runs") / "default")
        self.checkpoint_dir: Path | None = (
            Path(checkpoint_dir).resolve() if checkpoint_dir is not None else None
        )
        self.best_metric: float = float("inf")
        self.config_hash = hash_config(config)
        self._scaler = (
            torch.amp.GradScaler(self.device.type)
            if amp and self.device.type in {"cuda", "cpu"}
            else None
        )

    @abstractmethod
    def step(self, batch: dict) -> dict:
        """One optimisation step. Returns metrics dict."""
        ...

    @abstractmethod
    def validate(self, loader: DataLoader) -> float:
        """Return the validation metric (lower = better)."""
        ...

    def autocast(self) -> torch.amp.autocast | contextlib.nullcontext:
        """Return an autocast context for the current device, when AMP is on."""
        if self.amp:
            return torch.amp.autocast(device_type=self.device.type)
        return contextlib.nullcontext()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        *,
        epochs: int,
        patience: int = 10,
        resume: Path | str | None = None,
    ) -> dict:
        """Run the full training loop with early stopping and checkpointing.

        Args:
            train_loader: Training dataloader.
            val_loader: Optional validation dataloader.
            epochs: Maximum number of epochs.
            patience: Epochs without improvement before stopping.
            resume: Optional checkpoint to resume from.

        Returns
        -------
            Dict with final metrics.
        """
        start_epoch = 0
        if resume is not None:
            state = State.load(resume, expected_config_hash=self.config_hash)
            self.model.load_state_dict(state.model)
            if self.optimizer is not None and state.optimizer is not None:
                self.optimizer.load_state_dict(state.optimizer)
            start_epoch = state.epoch
            self.best_metric = state.metric
        no_improve = 0
        for epoch in range(start_epoch, start_epoch + epochs):
            self.model.train()
            epoch_metrics = self._run_epoch(train_loader, epoch)
            self.monitor.log(epoch=epoch, phase="train", **epoch_metrics)
            val_metric = float("inf")
            if val_loader is not None:
                val_metric = self.validate(val_loader)
                self.monitor.log(epoch=epoch, phase="val", metric=val_metric)
            if val_metric < self.best_metric:
                self.best_metric = val_metric
                no_improve = 0
                if self.checkpoint_dir is not None:
                    self._save(epoch, val_metric)
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
            if self.scheduler is not None:
                self.scheduler.step()
        return {"best": self.best_metric}

    def _run_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.model.train()
        running = 0.0
        count = 0
        for batch in loader:
            metrics = self.step(batch)
            running += float(metrics.get("loss", 0.0))
            count += 1
        return {"loss": running / max(count, 1)}

    def _save(self, epoch: int, metric: float) -> None:
        if self.checkpoint_dir is None:
            return
        state = State(
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict() if self.optimizer is not None else None,
            epoch=epoch,
            metric=metric,
            rng=None,
            config_hash=self.config_hash,
        )
        state.save(self.checkpoint_dir / "best.pt")

    def clip(self, params: list[torch.nn.Parameter]) -> None:
        """Apply gradient clipping if configured."""
        if self.grad_clip is not None and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, self.grad_clip)


__all__ = ["Trainer"]
