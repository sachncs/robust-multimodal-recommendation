"""Modality availability masks.

A mask is a 2-D binary array of shape ``(items, modalities)`` where ``1``
means "kept" and ``0`` means "missing". Every item is guaranteed to have at
least one kept modality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from morel.core.errors import Datum


@runtime_checkable
class Spec(Protocol):
    """Factory for mask arrays."""

    def sample(self, items: int, modalities: int, *, seed: int) -> np.ndarray:
        """Sample a binary mask of shape ``(items, modalities)``."""
        ...


@dataclass(frozen=True)
class Mask:
    """Immutable mask value object with the Mask Protocol."""

    data: np.ndarray

    def __post_init__(self) -> None:
        """Validate the mask data on construction."""
        if self.data.ndim != 2:
            raise Datum(f"mask must be 2-D, got {self.data.ndim}-D")
        if self.data.shape[0] == 0:
            raise Datum("mask has no items")
        if not np.all(np.isin(self.data, [0.0, 1.0])):
            raise Datum("mask values must be 0 or 1")
        if (self.data.sum(axis=1) == 0).any():
            raise Datum("mask rows must keep at least one modality")

    @property
    def items(self) -> int:
        """Return the number of items."""
        return int(self.data.shape[0])

    @property
    def modalities(self) -> int:
        """Return the number of modalities."""
        return int(self.data.shape[1])

    def kept(self, item: int) -> np.ndarray:
        """Return which modalities are kept for a given item."""
        kept: np.ndarray = self.data[item] > 0
        return kept

    def missing(self, item: int) -> np.ndarray:
        """Return which modalities are missing for a given item."""
        absent: np.ndarray = self.data[item] == 0
        return absent

    def numpy(self) -> np.ndarray:
        """Return the underlying array."""
        return self.data


def bernoulli(items: int, modalities: int, ratio: float, *, seed: int) -> Mask:
    """Independent Bernoulli masking with at-least-one repair.

    Args:
        items: Number of items.
        modalities: Number of modalities.
        ratio: Probability of masking per (item, modality) pair.
        seed: Random seed.

    Returns
    -------
        A ``Mask`` with shape ``(items, modalities)``.
    """
    if not 0.0 <= ratio <= 1.0:
        raise Datum(f"ratio must be in [0, 1], got {ratio}")
    if items <= 0 or modalities <= 0:
        raise Datum("items and modalities must be positive")
    rng = np.random.default_rng(seed)
    base = (rng.random((items, modalities)) > ratio).astype(np.float32)
    # Vectorized repair: rows with no kept modality get the first one set.
    rowsums = base.sum(axis=1)
    needs_repair = rowsums == 0
    if needs_repair.any():
        first_keep = np.argmax(base[needs_repair] == 1, axis=1)
        # If a row is entirely zero, force-keep modality 0.
        no_kept = base[needs_repair].sum(axis=1) == 0
        first_keep = np.where(no_kept, 0, first_keep)
        rows = np.where(needs_repair)[0]
        base[rows, first_keep] = 1.0
    return Mask(data=base)


def block(items: int, modalities: int, block_size: int, *, seed: int) -> Mask:
    """Block masking: contiguous modality spans are masked together.

    Args:
        items: Number of items.
        modalities: Number of modalities.
        block_size: Length of each block.
        seed: Random seed.

    Returns
    -------
        A ``Mask``.
    """
    if block_size <= 0 or block_size > modalities:
        raise Datum(f"block_size must be in [1, {modalities}], got {block_size}")
    rng = np.random.default_rng(seed)
    base = np.ones((items, modalities), dtype=np.float32)
    for i in range(items):
        start = int(rng.integers(0, modalities - block_size + 1))
        base[i, start : start + block_size] = 0.0
    return Mask(data=base)


def structured(pattern: np.ndarray) -> Mask:
    """Use a fixed pattern (e.g. ground-truth modality availability) as the mask.

    Args:
        pattern: 2-D binary array of shape ``(items, modalities)``.

    Returns
    -------
        A ``Mask``.
    """
    pattern = pattern.astype(np.float32, copy=False)
    return Mask(data=pattern)


def stack(masks: list[Mask]) -> np.ndarray:
    """Stack a list of masks into a 3-D array ``(len(masks), items, modalities)``."""
    if not masks:
        raise Datum("stack requires at least one mask")
    arrays = [m.numpy() for m in masks]
    return np.stack(arrays, axis=0)


__all__ = ["Mask", "Spec", "bernoulli", "block", "stack", "structured"]
