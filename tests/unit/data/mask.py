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
from morel.core.errors import Datum
from morel.data import MASKS, build_mask


class Checker:
    """Aggregated test methods for this module."""

    def strategies(self) -> None:
        assert set(MASKS) >= {"bernoulli", "block"}

    def every(self) -> None:
        """Completion is impossible for an item with nothing observed."""
        for ratio in (0.1, 0.5, 0.9):
            mask = build_mask("bernoulli", items=40, modalities=2, ratio=ratio, seed=0).to_numpy()
            assert (mask.sum(axis=1) > 0).all()

    def bernoulli(self) -> None:
        low = build_mask("bernoulli", items=200, modalities=3, ratio=0.1, seed=0).to_numpy()
        high = build_mask("bernoulli", items=200, modalities=3, ratio=0.9, seed=0).to_numpy()
        assert low.mean() > high.mean()

    def masking(self) -> None:
        first = build_mask("bernoulli", items=50, modalities=2, ratio=0.4, seed=3).to_numpy()
        second = build_mask("bernoulli", items=50, modalities=2, ratio=0.4, seed=3).to_numpy()
        assert np.array_equal(first, second)

    def block(self) -> None:
        with pytest.raises(Datum, match="at least 2 modalities"):
            build_mask("block", items=5, modalities=1, ratio=0.5, seed=0)

    def unknown(self) -> None:
        with pytest.raises(Datum, match="unknown masking kind"):
            build_mask("nope", items=5, modalities=2, ratio=0.5, seed=0)

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
