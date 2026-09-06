"""Masking must be selectable and driven by configuration.

``config.masking.kind``, ``ratio`` and ``seed`` were all inert:
``synthetic_dataset`` hardcoded ``bernoulli(items, 2, 0.4, seed=0)``. The
missing-modality pattern is the experimental condition for this method, so a
configured ratio that had no effect meant the run did not match its own record.
"""

from __future__ import annotations

import numpy as np
import pytest

from morel.app.experiment import synthetic_dataset
from morel.core.config import Masking
from morel.core.errors import ConfigError, DataError
from morel.data import MASKS


def test_strategies_are_registered() -> None:
    assert set(MASKS.available()) >= {"bernoulli", "block"}


@pytest.mark.parametrize("kind", ["bernoulli", "block"])
@pytest.mark.parametrize("ratio", [0.2, 0.5, 0.8])
def test_every_item_keeps_a_modality(kind: str, ratio: float) -> None:
    """Completion is impossible for an item with nothing observed."""
    mask = MASKS.create(kind, items=40, modalities=2, ratio=ratio, seed=0).to_numpy()
    assert (mask.sum(axis=1) > 0).all()


def test_bernoulli_ratio_changes_how_much_is_missing() -> None:
    low = MASKS.create("bernoulli", items=200, modalities=3, ratio=0.1, seed=0).to_numpy()
    high = MASKS.create("bernoulli", items=200, modalities=3, ratio=0.9, seed=0).to_numpy()
    assert low.mean() > high.mean()


def test_masking_is_reproducible_for_a_seed() -> None:
    first = MASKS.create("bernoulli", items=50, modalities=2, ratio=0.4, seed=3).to_numpy()
    second = MASKS.create("bernoulli", items=50, modalities=2, ratio=0.4, seed=3).to_numpy()
    assert np.array_equal(first, second)


def test_block_masking_needs_two_modalities() -> None:
    with pytest.raises(DataError, match="at least 2 modalities"):
        MASKS.create("block", items=5, modalities=1, ratio=0.5, seed=0)


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown masking strategy"):
        MASKS.create("nope", items=5, modalities=2, ratio=0.5, seed=0)


def test_synthetic_dataset_honours_the_masking_ratio() -> None:
    sparse = synthetic_dataset(80, 4, 2, 10, Masking(ratio=0.1))
    dense = synthetic_dataset(80, 4, 2, 10, Masking(ratio=0.9))
    assert sparse["mask"].mean() > dense["mask"].mean()


def test_synthetic_dataset_honours_the_masking_seed() -> None:
    first = synthetic_dataset(40, 4, 2, 10, Masking(seed=1))["mask"]
    second = synthetic_dataset(40, 4, 2, 10, Masking(seed=2))["mask"]
    assert not np.array_equal(first, second)
    again = synthetic_dataset(40, 4, 2, 10, Masking(seed=1))["mask"]
    assert np.array_equal(first, again)


def test_synthetic_dataset_honours_the_masking_kind() -> None:
    bern = synthetic_dataset(40, 4, 2, 10, Masking(kind="bernoulli", ratio=0.5, seed=0))["mask"]
    blk = synthetic_dataset(40, 4, 2, 10, Masking(kind="block", ratio=0.5, seed=0))["mask"]
    assert not np.array_equal(bern, blk)


def test_synthetic_dataset_defaults_to_the_shipped_masking() -> None:
    explicit = synthetic_dataset(40, 4, 2, 10, Masking())["mask"]
    implicit = synthetic_dataset(40, 4, 2, 10)["mask"]
    assert np.array_equal(explicit, implicit)
