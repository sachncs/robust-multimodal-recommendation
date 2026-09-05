"""Checkpoint save/load with config hash binding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from morel.core.errors import ConfigError


@dataclass
class State:
    """Trainer state snapshot for resume."""

    model: dict
    optimizer: dict | None
    epoch: int
    metric: float
    rng: dict | None
    config_hash: str
    extras: dict = field(default_factory=dict)

    def save(self, target: Path | str) -> None:
        """Atomically save the checkpoint."""
        path = Path(target).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "optimizer": self.optimizer,
            "epoch": self.epoch,
            "metric": self.metric,
            "rng": self.rng,
            "config_hash": self.config_hash,
            "extras": self.extras,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    @classmethod
    def load(cls, target: Path | str, *, expected_config_hash: str | None = None) -> "State":
        """Load a checkpoint, optionally verifying the config hash."""
        path = Path(target)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if expected_config_hash is not None and payload.get("config_hash") != expected_config_hash:
            raise ConfigError(
                f"checkpoint config hash mismatch: "
                f"got {payload.get('config_hash')}, expected {expected_config_hash}"
            )
        return cls(
            model=payload["model"],
            optimizer=payload.get("optimizer"),
            epoch=int(payload.get("epoch", 0)),
            metric=float(payload.get("metric", 0.0)),
            rng=payload.get("rng"),
            config_hash=str(payload.get("config_hash", "")),
            extras=dict(payload.get("extras", {})),
        )


def hash_config(config: object) -> str:
    """Stable SHA256 hash of a configuration object's public attributes."""
    if hasattr(config, "hash") and callable(config.hash):
        return config.hash()
    raw = json.dumps(config, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


__all__ = ["State", "hash_config"]
