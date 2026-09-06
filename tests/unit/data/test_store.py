"""Tests for morel.data.store."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from morel.core.errors import DataError
from morel.data.manifest import Manifest
from morel.data.store import load_graph, load_npz, save_graph, save_npz


class Checker:
    """Aggregated test methods for this module."""

    def save(tmp_path: Path) -> None:
        target = tmp_path / "x.npz"
        save_npz(target, a=np.arange(4, dtype=np.float32), b=np.ones(4))
        loaded = load_npz(target)
        assert np.array_equal(loaded["a"], np.arange(4))
        assert np.array_equal(loaded["b"], np.ones(4))

    def load(tmp_path: Path) -> None:
        target = tmp_path / "x.npz"
        m = Manifest(
            dataset="Beauty", version="1", code="c", seed=0, extractor="text", config_hash="abc"
        )
        save_npz(target, manifest_obj=m, a=np.arange(2, dtype=np.float32))
        assert load_npz(target).get("a") is not None

    def npz(tmp_path: Path) -> None:
        with pytest.raises(DataError):
            load_npz(tmp_path / "missing.npz")

    def graph(tmp_path: Path) -> None:
        target = tmp_path / "g.npz"
        g = sp.csr_matrix(np.eye(4, dtype=np.float32))
        save_graph(target, g)
        out = load_graph(target)
        assert out.shape == (4, 4)
        assert np.allclose(out.toarray(), np.eye(4))

    def missing(tmp_path: Path) -> None:
        with pytest.raises(DataError):
            load_graph(tmp_path / "missing.npz")

    def empty(tmp_path: Path) -> None:
        with pytest.raises(DataError):
            save_npz(tmp_path / "x.npz")
