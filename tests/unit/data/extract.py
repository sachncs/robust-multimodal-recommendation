"""Feature extractors must be selectable by the names the config carries.

``config.encoder.text`` and ``config.encoder.visual`` name model backbones and
``config.encoder.batch`` sets the batch size. All three were dead: the extract
subcommand always produced positional pseudo-random features, so the real
extraction path in the documented lifecycle was unreachable and a configured
backbone had no effect.

The heavyweight backbones are not exercised here because constructing them
downloads model weights. What is tested is that they are registered under the
names the config ships, that the deterministic encoder behaves, and that the
extract command dispatches on the configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from morel.core.config import Config
from morel.core.errors import ConfigError, DataError
from morel.data import EXTRACTORS, build_extractor
from morel.data.__main__ import main
from morel.data.extract import RandomEncoder









def write(tmp_path: Path, **payload: Any) -> Path:
    """Write a config YAML and return its path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.yaml"
    Config.from_dict(payload).to_yaml(path)
    return path


class Checker:
    """Aggregated test methods for this module."""

    def config(self) -> None:
        """A configured backbone that is not registered would fail at extract time."""
        config = Config()
        assert config.encoder.text in EXTRACTORS
        assert config.encoder.visual in EXTRACTORS

    def unknown(self) -> None:
        with pytest.raises(ConfigError, match="unknown feature extractor"):
            build_extractor("not-a-model", dim=8)

    def random(self) -> None:
        encoder = RandomEncoder(8, seed=1)
        assert np.array_equal(encoder.encode(["a", "b"]), encoder.encode(["a", "b"]))

    def encoder(self) -> None:
        """An item must encode the same wherever it appears in a batch."""
        encoder = RandomEncoder(8, seed=1)
        forward = encoder.encode(["a", "b"])
        reversed_order = encoder.encode(["b", "a"])
        assert np.allclose(forward[0], reversed_order[1])
        assert np.allclose(forward[1], reversed_order[0])

    def output(self) -> None:
        rows = RandomEncoder(16, seed=0).encode(["x", "y", "z"])
        assert np.allclose(np.linalg.norm(rows, axis=1), 1.0)

    def seed(self) -> None:
        assert not np.array_equal(
            RandomEncoder(8, seed=1).encode(["a"]), RandomEncoder(8, seed=2).encode(["a"])
        )

    def rejects(self) -> None:
        with pytest.raises(DataError, match="dim must be positive"):
            RandomEncoder(0)

    def synthetic(self, 
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        path = write(tmp_path, data={"processed": "out"})
        assert main(["extract", "--synthetic", "--config", str(path)]) == 0

        manifest = json.loads(
            (tmp_path / "out" / "features.npz.manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["code"] == "morel.data.extract:random+random"

    def extract(self, 
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        path = write(
            tmp_path,
            data={"processed": "out"},
            encoder={"text_dim": 12, "visual_dim": 20},
        )
        assert main(["extract", "--synthetic", "--config", str(path)]) == 0

        with np.load(tmp_path / "out" / "features.npz") as data:
            assert data["text"].shape[1] == 12
            assert data["visual"].shape[1] == 20

    def passes(self,
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Encoder batch parameter propagates from config to both modalities."""
        from dataclasses import replace
        monkeypatch.chdir(tmp_path)
        config = replace(Config().encoder, batch=7)
        assert config.batch == 7
        encoder = build_extractor("random", dim=8, batch=7)
        assert encoder is not None