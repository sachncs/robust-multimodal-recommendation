"""Shared fixtures and autouse seeding for all tests."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from tests.shared import build_path_graph, silent_monitor

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("MOREL_DATA_DIR", "/tmp/morel-test-data")


@pytest.fixture(autouse=True)
def deterministic_seed() -> None:
    """Make every test deterministic by default."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    yield


@pytest.fixture
def path3() -> sp.csr_matrix:
    """Three-node path graph: 0 - 1 - 2."""
    arr = np.array(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        dtype=np.float32,
    )
    return sp.csr_matrix(arr)


@pytest.fixture
def path5() -> sp.csr_matrix:
    """Five-node path graph: 0 - 1 - 2 - 3 - 4."""
    arr = np.zeros((5, 5), dtype=np.float32)
    for i in range(4):
        arr[i, i + 1] = 1
        arr[i + 1, i] = 1
    return sp.csr_matrix(arr)


@pytest.fixture
def bipartite_5x10() -> sp.csr_matrix:
    """5 users, 10 items."""
    arr = np.array(
        [
            [1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 1, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    return sp.csr_matrix(arr)


@pytest.fixture
def visual_text_features() -> dict[str, np.ndarray]:
    """Visual + text features for 5 items."""
    rng = np.random.default_rng(0)
    return {
        "visual": rng.normal(size=(5, 4)).astype(np.float32),
        "text": rng.normal(size=(5, 2)).astype(np.float32),
    }


@pytest.fixture
def full_mask() -> np.ndarray:
    """All-ones mask for 5 items x 2 modalities."""
    return np.ones((5, 2), dtype=np.float32)


@pytest.fixture
def tmp_manifest(tmp_path) -> "Path":  # type: ignore[name-defined]
    """A directory pre-created for manifest artifacts."""
    from pathlib import Path

    d = tmp_path / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def path_graph_factory():
    """Factory exposing :func:`build_path_graph` for parametric tests."""
    return build_path_graph


@pytest.fixture
def silent_monitor_factory():
    """Factory exposing :func:`silent_monitor` for parametric tests."""
    return silent_monitor
