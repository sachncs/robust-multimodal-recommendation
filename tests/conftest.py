"""Shared fixtures, autouse seeding, and collection rules for all tests.

Fixtures are defined here so every test can use them. The
``pytest_collection_modifyitems`` hook removes any collected test that is
not a method of a ``Checker``/``Spec`` class, so standalone helper
functions in test files and production modules are never run.
"""

from __future__ import annotations

import os
import random
from typing import Any, Callable

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from tests.shared import build_path_graph, silent_monitor

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("MOREL_DATA_DIR", "/tmp/morel-test-data")


@pytest.fixture(autouse=True)
def deterministic_seed() -> Any:
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
def path_graph_factory() -> Callable[[int], sp.csr_matrix]:
    """Factory that builds a path graph of ``n`` nodes."""
    return build_path_graph


@pytest.fixture
def silent_monitor_factory() -> Callable[..., object]:
    """Factory that builds a no-op training monitor."""
    return silent_monitor


# ---------------------------------------------------------------------------
# Collection rules
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Drop any collected test that is not a method of a test class."""
    from _pytest.python import Class  # type: ignore[attr-defined]

    keep: list[pytest.Item] = []
    for item in items:
        # A test method has a parent that is a Class instance.
        parent = item.parent
        if isinstance(parent, Class) and parent.name in {"Checker", "Spec"}:
            keep.append(item)
    items[:] = keep
