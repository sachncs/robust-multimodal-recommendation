"""Recommendation trainer: BPR with strict negatives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from morel.recommend.bpr import bpr as bpr_loss
from morel.recommend.bpr import negatives as sample_negatives
from morel.train.checkpoint import hash_config
from morel.train.monitor import Monitor
from morel.train.trainer import Trainer


@dataclass
class RecommendationConfig:
    """Configuration for recommendation training."""

    grad_clip: float = 1.0

    def hash(self) -> str:
        import hashlib
        import json

        from dataclasses import asdict

        raw = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Recommendation(Trainer):
    """BPR trainer with strict negatives and on-the-fly scoring."""

    def __init__(
        self,
        model,
        config,
        *,
        ui_graph,
        negatives_count: int = 1,
        seed: int = 0,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        monitor: Monitor | None = None,
        checkpoint_dir=None,
        device: str | torch.device | None = None,
        amp: bool = False,
    ) -> None:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
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

    def step(self, batch: dict) -> dict:
        """One BPR step on a user batch."""
        users = batch["users"].to(self.device)
        pos = batch["positive"].to(self.device)
        neg = batch["negative"].to(self.device)
        self.optimizer.zero_grad()
        scores = self.model(users, torch.arange(self.items, device=self.device), self.ui_graph)
        pos_scores = scores[torch.arange(users.shape[0], device=self.device), pos]
        neg_scores = scores[torch.arange(users.shape[0], device=self.device), neg]
        loss = bpr_loss(pos_scores, neg_scores)
        loss.backward()
        self.clip(list(self.model.parameters()))
        self.optimizer.step()
        return {"loss": float(loss.item())}

    def validate(self, loader: DataLoader) -> float:
        """Return the BPR loss on the validation loader."""
        self.model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                users = batch["users"].to(self.device)
                pos = batch["positive"].to(self.device)
                neg = batch["negative"].to(self.device)
                scores = self.model(users, torch.arange(self.items, device=self.device), self.ui_graph)
                pos_scores = scores[torch.arange(users.shape[0], device=self.device), pos]
                neg_scores = scores[torch.arange(users.shape[0], device=self.device), neg]
                loss = bpr_loss(pos_scores, neg_scores)
                total += float(loss.item())
                count += 1
        return total / max(count, 1)


__all__ = ["Recommendation", "RecommendationConfig"]
