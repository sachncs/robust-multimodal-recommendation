"""Tests for morel.train.checkpoint and morel.train.monitor."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from morel.core.errors import ConfigError, ModelError
from morel.train.checkpoint import State, hash_config, safe_load, unsafe_load
from morel.train.monitor import Monitor


class CheckpointMarker:
    """Module-level marker class used to verify unsafe_load roundtrips."""


class Checker:
    """Aggregated test methods for this module."""

    def roundtrip(self, tmp_path: Path) -> None:
        model = nn.Linear(4, 2)
        state = State(
            model=model.state_dict(),
            optimizer=None,
            epoch=3,
            metric=0.5,
            rng=None,
            config_hash="abc",
        )
        path = tmp_path / "c.pt"
        state.save(path)
        loaded = State.load(path)
        assert loaded.epoch == 3
        assert loaded.metric == 0.5
        assert loaded.config_hash == "abc"

    def mismatch(self, tmp_path: Path) -> None:
        model = nn.Linear(2, 1)
        state = State(
            model=model.state_dict(), optimizer=None, epoch=0, metric=0.0, rng=None, config_hash="a"
        )
        path = tmp_path / "c.pt"
        state.save(path)
        with __import__("pytest").raises(ConfigError):
            State.load(path, expected_config_hash="b")

    def dataclass(self) -> None:

        from morel.core.config import Config

        h = hash_config(Config())
        assert len(h) == 64

    def read(self, tmp_path: Path) -> None:
        mon = Monitor(tmp_path)
        mon.log(step=1, loss=0.1)
        mon.log(step=2, loss=0.05)
        latest = mon.latest()
        assert latest is not None
        assert latest["step"] == 2
        assert latest["loss"] == 0.05

    def none(self, tmp_path: Path) -> None:
        mon = Monitor(tmp_path)
        assert mon.latest() is None

    def payload(self, tmp_path: Path) -> None:
        """safe_load refuses payloads that contain a __reduce__ exploit class."""

        class Exploit:
            def __reduce__(self):  # pragma: no cover - never executed
                return (exec, ("print('pwned')",))

        bad_path = tmp_path / "exploit.pt"
        torch.save({"model": {"evil": Exploit()}}, bad_path)
        with pytest.raises(ModelError):
            safe_load(bad_path)

    def keys(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "bad.pt"
        torch.save({"model": {}, "unexpected": True}, bad_path)
        with pytest.raises(ModelError):
            safe_load(bad_path)

    def exploit(self, tmp_path: Path) -> None:
        """unsafe_load is the explicit opt-in for non-tensor payloads."""
        payload_path = tmp_path / "ok.pt"
        torch.save(
            {"model": {"x": torch.zeros(1)}, "obj": CheckpointMarker},
            payload_path,
        )
        payload = unsafe_load(payload_path)
        assert "obj" in payload

    def dict(self, tmp_path: Path) -> None:
        p = tmp_path / "scalar.pt"
        torch.save(42, p)
        with pytest.raises(ModelError):
            safe_load(p)

    def raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            safe_load(tmp_path / "nope.pt")