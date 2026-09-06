"""Tests for morel.retrieve.anchor."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from morel.core.errors import Net, Shape
from morel.retrieve.anchor import batch, query


@pytest.fixture
def setup() -> tuple[dict[str, np.ndarray], np.ndarray, sp.csr_matrix]:
    g = sp.csr_matrix(np.eye(5, dtype=np.float32))
    features = {
        "visual": np.eye(5, dtype=np.float32),
        "text": np.eye(5, dtype=np.float32),
    }
    mask = np.ones((5, 2), dtype=np.float32)
    return features, mask, g


class Checker:
    """Aggregated test methods for this module."""

    def self(self, setup) -> None:
        features, mask, _ = setup
        anchors = query(0, "visual", features, mask, top=2)
        assert 0 not in anchors
        assert anchors.tolist() == [1, 2]

    def modality(self, setup) -> None:
        features, mask, _ = setup
        assert query(0, "missing", features, mask, top=2).tolist() == []

    def range(self, setup) -> None:
        features, mask, _ = setup
        with pytest.raises(Net):
            query(99, "visual", features, mask, top=2)

    def top(self, setup) -> None:
        features, mask, _ = setup
        with pytest.raises(Net):
            query(0, "visual", features, mask, top=0)

    def list(self, setup) -> None:
        features, mask, _ = setup
        out = batch([0, 1, 2], "visual", features, mask, top=2)
        assert len(out) == 3

    def norm(self) -> None:
        features = {"visual": np.zeros((3, 4), dtype=np.float32)}
        mask = np.ones((3, 1), dtype=np.float32)
        out = query(0, "visual", features, mask, top=2)
        assert out.tolist() == []

    def mask(self) -> None:
        features = {"a": np.zeros((3, 4), dtype=np.float32)}
        mask = np.ones((4, 1), dtype=np.float32)
        with pytest.raises(Shape):
            query(0, "a", features, mask, top=2)