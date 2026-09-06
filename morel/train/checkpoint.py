"""Checkpoint save/load with config hash binding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from morel.core.errors import ConfigError, ModelError

ALLOWED = {"model", "optimizer", "epoch", "metric", "rng", "config_hash", "extras"}


def load(target: Path | str) -> dict[str, Any]:
    """Load a checkpoint payload safely.

    Uses ``weights_only=True`` to disable arbitrary pickle deserialization
    and validates the payload shape against the morel checkpoint contract.

    Args:
        target: Path to the checkpoint file.

    Returns
    -------
        The validated payload dict.

    Raises
    ------
        FileNotFoundError: If the file does not exist.
        ModelError: If the payload shape does not match the morel checkpoint contract.
    """
    path = Path(target)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelError(f"checkpoint at {path} could not be loaded safely: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelError(f"checkpoint at {path} must be a dict, got {type(payload).__name__}")
    unknown = set(payload.keys()) - ALLOWED
    if unknown:
        raise ModelError(f"checkpoint at {path} has unknown keys: {sorted(unknown)}")
    return payload


def unsafe_load(target: Path | str) -> dict[str, Any]:
    """Load a checkpoint allowing arbitrary pickle deserialization.

    Use only for trusted, in-house checkpoints that contain non-Tensor
    state (e.g., custom optimizer buffers from older versions).
    """
    path = Path(target)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ModelError(f"checkpoint at {path} must be a dict, got {type(payload).__name__}")
    return payload


@dataclass
class State:
    """Trainer state snapshot for resume."""

    model: dict[str, Any]
    optimizer: dict[str, Any] | None
    epoch: int
    metric: float
    rng: dict[str, Any] | None
    config_hash: str
    extras: dict[str, Any] = field(default_factory=dict)

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
    def load(cls, target: Path | str, *, expected_config_hash: str | None = None) -> State:
        """Load a checkpoint, optionally verifying the config hash."""
        payload = load(target)
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
        digest: str = config.hash()
        return digest
    raw = json.dumps(config, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


__all__ = ["State", "hash_config", "load", "unsafe_load"]
