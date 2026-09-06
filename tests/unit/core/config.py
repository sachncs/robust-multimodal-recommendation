"""Tests for morel.core.config."""

from __future__ import annotations

import pytest

from morel.core.config import Config
from morel.core.errors import ConfigError


class Checker:
    """Aggregated test methods for this module."""

    def default(self) -> None:
        c = Config()
        c.validate()

    def stable(self) -> None:
        a = Config().hash()
        b = Config().hash()
        assert a == b

    def changes(self) -> None:
        a = Config()
        b = Config(seed=43)
        assert a.hash() != b.hash()

    def invalid(self) -> None:
        with pytest.raises(ConfigError):
            Config(seed=-1).validate()

    def mask(self) -> None:
        from dataclasses import replace

        bad = replace(Config(), masking=replace(Config().masking, ratio=2.0))
        with pytest.raises(ConfigError):
            bad.validate()

    def to(self, tmp_path) -> None:
        c = Config()
        path = tmp_path / "config.yaml"
        c.save(path)
        loaded = Config.from_yaml(path)
        assert loaded.hash() == c.hash()

    def rejects(self) -> None:
        with pytest.raises(ConfigError):
            Config.from_dict({"seed": 0, "totally_made_up": True})

    def env(self, monkeypatch) -> None:
        monkeypatch.setenv("MOREL_SEED", "123")
        c = Config.load_env()
        assert c.seed == 123