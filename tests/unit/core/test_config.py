"""Tests for morel.core.config."""

from __future__ import annotations

import pytest

from morel.core.config import Config
from morel.core.errors import ConfigError


def test_default_config_validates() -> None:
    c = Config()
    c.validate()


def test_hash_is_stable() -> None:
    a = Config().hash()
    b = Config().hash()
    assert a == b


def test_hash_changes_with_value() -> None:
    a = Config()
    b = Config(seed=43)
    assert a.hash() != b.hash()


def test_invalid_seed_raises() -> None:
    with pytest.raises(ConfigError):
        Config(seed=-1).validate()


def test_invalid_mask_ratio_raises() -> None:
    from dataclasses import replace

    bad = replace(Config(), masking=replace(Config().masking, ratio=2.0))
    with pytest.raises(ConfigError):
        bad.validate()


def test_to_yaml_roundtrip(tmp_path) -> None:
    c = Config()
    path = tmp_path / "config.yaml"
    c.to_yaml(path)
    loaded = Config.from_yaml(path)
    assert loaded.hash() == c.hash()


def test_from_dict_rejects_unknown_key() -> None:
    with pytest.raises(ConfigError):
        Config.from_dict({"seed": 0, "totally_made_up": True})


def test_from_env_overrides_seed(monkeypatch) -> None:
    monkeypatch.setenv("MOREL_SEED", "123")
    c = Config.from_env()
    assert c.seed == 123
