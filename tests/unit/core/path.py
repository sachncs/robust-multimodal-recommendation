"""Tests for morel.core.path."""

from __future__ import annotations

import pytest

from morel.core.errors import ConfigError
from morel.core.path import checkpoints, features, graphs, manifest, processed, root, runs


class Checker:
    """Aggregated test methods for this module."""

    def root(tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("MOREL_DATA_DIR", str(tmp_path))
        assert root() == tmp_path.resolve()

    def processed() -> None:
        with pytest.raises(ConfigError):
            processed("../escape")
        with pytest.raises(ConfigError):
            processed("")

    def manifest(tmp_path) -> None:
        target = tmp_path / "x.npz"
        target.write_text("x")
        assert manifest(target).name == "x.npz.manifest.json"

    def checkpoints(tmp_path) -> None:
        with pytest.raises(ConfigError):
            checkpoints("")

    def features(tmp_path) -> None:
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("MOREL_DATA_DIR", str(tmp_path))
        p = processed("Beauty")
        assert (p / "features").parent == p
        assert features("Beauty") == p / "features"
        assert graphs("Beauty") == p / "graphs"
        monkeypatch.undo()

    def runs() -> None:
        with pytest.raises(ConfigError):
            runs("")
