"""Tests for morel.core.path."""

from __future__ import annotations

import pytest

from morel.core.errors import Cfg
from morel.core.path import checkpoints, features, graphs, manifest, processed, root, runs


class Checker:
    """Aggregated test methods for this module."""

    def root(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("MOREL_DATA_DIR", str(tmp_path))
        assert root() == tmp_path.resolve()

    def processed(self) -> None:
        with pytest.raises(Cfg):
            processed("../escape")
        with pytest.raises(Cfg):
            processed("")

    def manifest(self, tmp_path) -> None:
        target = tmp_path / "x.npz"
        target.write_text("x")
        assert manifest(target).name == "x.npz.manifest.json"

    def checkpoints(self, tmp_path) -> None:
        with pytest.raises(Cfg):
            checkpoints("")

    def features(self, tmp_path) -> None:
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("MOREL_DATA_DIR", str(tmp_path))
        p = processed("Beauty")
        assert (p / "features").parent == p
        assert features("Beauty") == p / "features"
        assert graphs("Beauty") == p / "graphs"
        monkeypatch.undo()

    def runs(self) -> None:
        with pytest.raises(Cfg):
            runs("")