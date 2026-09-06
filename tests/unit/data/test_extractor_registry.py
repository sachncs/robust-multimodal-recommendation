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
from morel.data import EXTRACTORS
from morel.data.__main__ import main
from morel.data.extract import RandomEncoder


def test_config_default_encoders_are_registered() -> None:
    """A configured backbone that is not registered would fail at extract time."""
    config = Config()
    assert config.encoder.text in EXTRACTORS
    assert config.encoder.visual in EXTRACTORS


def test_unknown_extractor_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown feature extractor"):
        EXTRACTORS.create("not-a-model", dim=8)


def test_random_encoder_is_deterministic() -> None:
    encoder = RandomEncoder(8, seed=1)
    assert np.array_equal(encoder.encode(["a", "b"]), encoder.encode(["a", "b"]))


def test_random_encoder_depends_on_the_input_not_the_position() -> None:
    """An item must encode the same wherever it appears in a batch."""
    encoder = RandomEncoder(8, seed=1)
    forward = encoder.encode(["a", "b"])
    reversed_order = encoder.encode(["b", "a"])
    assert np.allclose(forward[0], reversed_order[1])
    assert np.allclose(forward[1], reversed_order[0])


def test_random_encoder_output_is_l2_normalized() -> None:
    rows = RandomEncoder(16, seed=0).encode(["x", "y", "z"])
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0)


def test_random_encoder_seed_changes_the_output() -> None:
    assert not np.array_equal(
        RandomEncoder(8, seed=1).encode(["a"]), RandomEncoder(8, seed=2).encode(["a"])
    )


def test_random_encoder_rejects_a_non_positive_dim() -> None:
    with pytest.raises(DataError, match="dim must be positive"):
        RandomEncoder(0)


def write_config(tmp_path: Path, **payload: Any) -> Path:
    """Write a config YAML and return its path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.yaml"
    Config.from_dict(payload).to_yaml(path)
    return path


def test_synthetic_extract_records_the_encoders_it_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = write_config(tmp_path, data={"processed": "out"})
    assert main(["extract", "--synthetic", "--config", str(path)]) == 0

    manifest = json.loads(
        (tmp_path / "out" / "features.npz.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["code"] == "morel.data.extract:random+random"


def test_extract_honours_the_configured_feature_widths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = write_config(
        tmp_path,
        data={"processed": "out"},
        encoder={"text_dim": 12, "visual_dim": 20},
    )
    assert main(["extract", "--synthetic", "--config", str(path)]) == 0

    with np.load(tmp_path / "out" / "features.npz") as data:
        assert data["text"].shape[1] == 12
        assert data["visual"].shape[1] == 20


def test_extract_passes_the_configured_batch_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    seen: list[int] = []
    real = EXTRACTORS.get("random")

    def spy(*, dim: int, batch: int = 64, seed: int = 0) -> Any:
        seen.append(batch)
        return real(dim=dim, batch=batch, seed=seed)

    EXTRACTORS.register("random", spy, replace=True)
    try:
        path = write_config(tmp_path, data={"processed": "out"}, encoder={"batch": 7})
        assert main(["extract", "--synthetic", "--config", str(path)]) == 0
    finally:
        EXTRACTORS.register("random", real, replace=True)

    assert seen == [7, 7], f"expected the configured batch for both modalities, got {seen}"
