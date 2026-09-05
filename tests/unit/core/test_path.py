"""Tests for morel.core.path."""

from __future__ import annotations

import pytest

from morel.core.errors import ConfigError
from morel.core.path import checkpoints, features, graphs, manifest, processed, raw, root, runs


def test_root_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MOREL_DATA_DIR", str(tmp_path))
    assert root() == tmp_path.resolve()


def test_processed_validates_name() -> None:
    with pytest.raises(ConfigError):
        processed("../escape")
    with pytest.raises(ConfigError):
        processed("")


def test_manifest_sidecar(tmp_path) -> None:
    target = tmp_path / "x.npz"
    target.write_text("x")
    assert manifest(target).name == "x.npz.manifest.json"


def test_checkpoints_rejects_empty(tmp_path) -> None:
    with pytest.raises(ConfigError):
        checkpoints("")


def test_features_and_graphs(tmp_path) -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("MOREL_DATA_DIR", str(tmp_path))
    p = processed("Beauty")
    assert (p / "features").parent == p
    assert features("Beauty") == p / "features"
    assert graphs("Beauty") == p / "graphs"
    monkeypatch.undo()


def test_runs_requires_id() -> None:
    with pytest.raises(ConfigError):
        runs("")
