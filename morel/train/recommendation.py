"""Recommendation trainer: BPR with strict negatives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from morel.recommend.bpr import bpr as bpr_loss
from morel.train.monitor import Monitor
from morel.train.trainer import Trainer


@dataclass
class RankCfg:
    """Configuration for recommendation training."""

    grad_clip: float = 1.0


class Recommendation(Trainer):
    """BPR trainer with strict negatives and on-the-fly scoring."""

    def __init__(
        self,
        model: nn.Module,
        config: RankCfg,
        *,
        ui_graph: sp.csr_matrix,
        negatives_count: int = 1,
        seed: int = 0,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        monitor: Monitor | None = None,
        checkpoint_dir: Path | str | None = None,
        device: str | torch.device | None = None,
        amp: bool = False,
    ) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        super().__init__(
            model=model,
            optimizer=optimizer,
            loss=None,
            config=config,
            monitor=monitor,
            checkpoint_dir=checkpoint_dir,
            grad_clip=config.grad_clip if hasattr(config, "grad_clip") else 1.0,
            device=device,
            amp=amp,
        )
        self.ui_graph = ui_graph
        self.users = ui_graph.shape[0]
        self.items = ui_graph.shape[1]
        self.negatives_count = negatives_count
        self.seed = seed
        self.negatives_matrix: np.ndarray | None = None

    def step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """One BPR step on a user batch."""
        users = batch["users"].to(self.device)
        pos = batch["positive"].to(self.device)
        neg = batch["negative"].to(self.device)
        self.optimizer.zero_grad()
        scores = self.model(users, torch.arange(self.items, device=self.device), self.ui_graph)
        pos_scores = scores[torch.arange(users.shape[0], device=self.device), pos]
        neg_scores = scores[torch.arange(users.shape[0], device=self.device), neg]
        loss = bpr_loss(pos_scores, neg_scores)
        loss.backward()  # type: ignore[no-untyped-call]  # torch stubs leave this untyped
        self.clip(list(self.model.parameters()))
        self.optimizer.step()
        return {"loss": float(loss.item())}

    def validate(self, loader: DataLoader[Any]) -> float:
        """Return the BPR loss on the validation loader."""
        self.model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                users = batch["users"].to(self.device)
                pos = batch["positive"].to(self.device)
                neg = batch["negative"].to(self.device)
                scores = self.model(
                    users, torch.arange(self.items, device=self.device), self.ui_graph
                )
                pos_scores = scores[torch.arange(users.shape[0], device=self.device), pos]
                neg_scores = scores[torch.arange(users.shape[0], device=self.device), neg]
                loss = bpr_loss(pos_scores, neg_scores)
                total += float(loss.item())
                count += 1
        return total / max(count, 1)


__all__ = ["RankCfg", "Recommendation"]
