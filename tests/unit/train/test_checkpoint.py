"""Tests for morel.train.checkpoint and morel.train.monitor."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from morel.core.errors import ConfigError
from morel.train.checkpoint import State, hash_config
from morel.train.monitor import Monitor


def test_state_roundtrip(tmp_path: Path) -> None:
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


def test_state_rejects_hash_mismatch(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    state = State(model=model.state_dict(), optimizer=None, epoch=0, metric=0.0, rng=None, config_hash="a")
    path = tmp_path / "c.pt"
    state.save(path)
    with __import__("pytest").raises(ConfigError):
        State.load(path, expected_config_hash="b")


def test_hash_config_for_dataclass() -> None:
    from dataclasses import dataclass

    from morel.core.config import Config

    h = hash_config(Config())
    assert len(h) == 64


def test_monitor_append_and_read(tmp_path: Path) -> None:
    mon = Monitor(tmp_path)
    mon.log(step=1, loss=0.1)
    mon.log(step=2, loss=0.05)
    latest = mon.latest()
    assert latest is not None
    assert latest["step"] == 2
    assert latest["loss"] == 0.05


def test_monitor_empty_returns_none(tmp_path: Path) -> None:
    mon = Monitor(tmp_path)
    assert mon.latest() is None
