"""Tests for morel.data.manifest."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from morel.core.errors import DataError
from morel.data.manifest import Manifest, checksum, load, path_for, save


def test_save_load_roundtrip(tmp_path: Path) -> None:
    artifact = tmp_path / "x.npz"
    artifact.write_text("hello")
    m = Manifest(
        dataset="Beauty",
        version="1",
        code="abc123",
        seed=0,
        extractor="text",
        config_hash="deadbeef",
    )
    sidecar = save(artifact, m)
    assert sidecar.exists()
    loaded = load(artifact)
    assert loaded.dataset == "Beauty"
    assert loaded.seed == 0
    assert loaded.schema == 1


def test_load_enforces_config_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "x.npz"
    artifact.write_text("hello")
    m = Manifest(
        dataset="Beauty",
        version="1",
        code="abc",
        seed=0,
        extractor="text",
        config_hash="hash1",
    )
    save(artifact, m)
    with pytest.raises(DataError):
        load(artifact, expected_config_hash="hash2")


def test_load_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(DataError):
        load(tmp_path / "missing.npz")


def test_checksum(tmp_path: Path) -> None:
    p = tmp_path / "x"
    p.write_text("hello world")
    assert checksum(p) == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_path_for(tmp_path: Path) -> None:
    p = tmp_path / "data.npz"
    assert path_for(p).name == "data.npz.manifest.json"
