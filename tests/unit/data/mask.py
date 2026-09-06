
import numpy as np
import pytest

from morel.core.errors import DataError
from morel.data.mask import Mask, bernoulli, block, stack, structured


class Checker:
    """Aggregated test methods for this module."""

    def bernoulli() -> None:
        m = bernoulli(20, 3, 0.4, seed=0)
        assert m.to_numpy().shape == (20, 3)
        assert m.to_numpy().dtype == np.float32
        assert m.items == 20
        assert m.modalities == 3

    def at() -> None:
        m = bernoulli(50, 1, 1.0, seed=1).to_numpy()
        assert (m.sum(axis=1) == 1).all()

    def invalid() -> None:
        with pytest.raises(DataError):
            bernoulli(5, 3, 1.5, seed=0)

    def seed() -> None:
        a = bernoulli(30, 4, 0.3, seed=42).to_numpy()
        b = bernoulli(30, 4, 0.3, seed=42).to_numpy()
        assert np.array_equal(a, b)

    def block() -> None:
        m = block(10, 4, 2, seed=0).to_numpy()
        assert (m.sum(axis=1) == 2).all()

    def structured() -> None:
        pattern = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        m = structured(pattern)
        assert m.items == 3
        assert m.modalities == 2

    def mask() -> None:
        with pytest.raises(DataError):
            Mask(data=np.array([[0, 0]], dtype=np.float32))

    def stack() -> None:
        a = bernoulli(4, 2, 0.5, seed=0)
        b = bernoulli(4, 2, 0.5, seed=1)
        c = bernoulli(4, 2, 0.5, seed=2)
        s = stack([a, b, c])
        assert s.shape == (3, 4, 2)
