"""Masking must be selectable and driven by configuration.

``config.masking.kind``, ``ratio`` and ``seed`` were all inert:
``synthetic`` hardcoded ``bernoulli(items, 2, 0.4, seed=0)``. The
missing-modality pattern is the experimental condition for this method, so a
configured ratio that had no effect meant the run did not match its own record.
"""

from __future__ import annotations

import numpy as np
import pytest

from morel.app.experiment import synthetic
from morel.core.config import Masking
from morel.core.errors import Cfg, Datum
from morel.data import MASKS


class Checker:
    """Aggregated test methods for this module."""

    def strategies(self) -> None:
        assert set(MASKS.available()) >= {"bernoulli", "block"}

    def every(self, kind: str, ratio: float) -> None:
        """Completion is impossible for an item with nothing observed."""
        mask = MASKS.create(kind, items=40, modalities=2, ratio=ratio, seed=0).to_numpy()
        assert (mask.sum(axis=1) > 0).all()

    def bernoulli(self) -> None:
        low = MASKS.create("bernoulli", items=200, modalities=3, ratio=0.1, seed=0).to_numpy()
        high = MASKS.create("bernoulli", items=200, modalities=3, ratio=0.9, seed=0).to_numpy()
        assert low.mean() > high.mean()

    def masking(self) -> None:
        first = MASKS.create("bernoulli", items=50, modalities=2, ratio=0.4, seed=3).to_numpy()
        second = MASKS.create("bernoulli", items=50, modalities=2, ratio=0.4, seed=3).to_numpy()
        assert np.array_equal(first, second)

    def block(self) -> None:
        with pytest.raises(Datum, match="at least 2 modalities"):
            MASKS.create("block", items=5, modalities=1, ratio=0.5, seed=0)

    def unknown(self) -> None:
        with pytest.raises(Cfg, match="unknown masking strategy"):
            MASKS.create("nope", items=5, modalities=2, ratio=0.5, seed=0)

    def synthetic(self) -> None:
        sparse = synthetic(80, 4, 2, 10, Masking(ratio=0.1))
        dense = synthetic(80, 4, 2, 10, Masking(ratio=0.9))
        assert sparse["mask"].mean() > dense["mask"].mean()

    def dataset(self) -> None:
        first = synthetic(40, 4, 2, 10, Masking(seed=1))["mask"]
        second = synthetic(40, 4, 2, 10, Masking(seed=2))["mask"]
        assert not np.array_equal(first, second)
        again = synthetic(40, 4, 2, 10, Masking(seed=1))["mask"]
        assert np.array_equal(first, again)

    def honours(self) -> None:
        bern = synthetic(40, 4, 2, 10, Masking(kind="bernoulli", ratio=0.5, seed=0))["mask"]
        blk = synthetic(40, 4, 2, 10, Masking(kind="block", ratio=0.5, seed=0))["mask"]
        assert not np.array_equal(bern, blk)

    def defaults(self) -> None:
        explicit = synthetic(40, 4, 2, 10, Masking())["mask"]
        implicit = synthetic(40, 4, 2, 10)["mask"]
        assert np.array_equal(explicit, implicit)


# --- merged from tests/unit/data/test_mask.py ---


import numpy as np
import pytest

from morel.core.errors import Datum
from morel.data.mask import Mask, bernoulli, block, stack, structured


class Checker:
    """Aggregated test methods for this module."""

    def bernoulli(self) -> None:
        m = bernoulli(20, 3, 0.4, seed=0)
        assert m.to_numpy().shape == (20, 3)
        assert m.to_numpy().dtype == np.float32
        assert m.items == 20
        assert m.modalities == 3

    def at(self) -> None:
        m = bernoulli(50, 1, 1.0, seed=1).to_numpy()
        assert (m.sum(axis=1) == 1).all()

    def invalid(self) -> None:
        with pytest.raises(Datum):
            bernoulli(5, 3, 1.5, seed=0)

    def seed(self) -> None:
        a = bernoulli(30, 4, 0.3, seed=42).to_numpy()
        b = bernoulli(30, 4, 0.3, seed=42).to_numpy()
        assert np.array_equal(a, b)

    def block(self) -> None:
        m = block(10, 4, 2, seed=0).to_numpy()
        assert (m.sum(axis=1) == 2).all()

    def structured(self) -> None:
        pattern = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        m = structured(pattern)
        assert m.items == 3
        assert m.modalities == 2

    def mask(self) -> None:
        with pytest.raises(Datum):
            Mask(data=np.array([[0, 0]], dtype=np.float32))

    def stack(self) -> None:
        a = bernoulli(4, 2, 0.5, seed=0)
        b = bernoulli(4, 2, 0.5, seed=1)
        c = bernoulli(4, 2, 0.5, seed=2)
        s = stack([a, b, c])
        assert s.shape == (3, 4, 2)