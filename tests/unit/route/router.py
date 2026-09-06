"""Tests for morel.route."""

from __future__ import annotations

import pytest
import torch

from morel.core.errors import ConfigError
from morel.route import Dense, Fixed, Gumbel, Top, Weights, build


class Checker:
    """Aggregated test methods for this module."""

    def routing(self) -> None:
        r = Dense(dim=8, k=5, tau=1.0)
        out = r(torch.randn(2, 8))
        assert out.probs.shape == (2, 5)
        assert torch.allclose(out.probs.sum(-1), torch.ones(2), atol=1e-5)

    def one(self) -> None:
        r = Top(dim=8, k=10, p=3, tau=0.5)
        out = r(torch.randn(4, 8), training=True)
        assert out.probs.shape == (4, 10)
        assert torch.allclose(out.probs.sum(-1), torch.ones(4), atol=1e-5)

    def sparse(self) -> None:
        r = Top(dim=8, k=10, p=3, tau=0.5)
        out = r(torch.randn(4, 8), training=False)
        nonzero = (out.probs > 0).sum(-1)
        assert (nonzero == 3).all()

    def raises(self) -> None:
        with pytest.raises(ValueError, match="must be <= k"):
            Top(dim=8, k=5, p=10)

    def tau(self) -> None:
        with pytest.raises(ValueError, match="tau must be positive"):
            Top(dim=8, k=5, p=2, tau=0.0)

    def to(self) -> None:
        r = Gumbel(dim=8, k=10, tau=0.5)
        out = r(torch.randn(4, 8), training=True)
        assert torch.allclose(out.probs.sum(-1), torch.ones(4), atol=1e-5)

    def factory(self) -> None:
        r = build("top", dim=8, k=10, p=3, tau=0.5)
        assert isinstance(r, Top)
        r = build("dense", dim=8, k=10, p=3, tau=0.5)
        assert isinstance(r, Dense)

    def unknown(self) -> None:
        with pytest.raises(ConfigError, match="unknown router"):
            build("nope", dim=8, k=10, p=3, tau=0.5)

    def shape(self) -> None:
        w = Weights(probs=torch.zeros(2, 5))
        assert w.shape == (2, 5)

    def index(self) -> None:
        r = Fixed(k=5)
        with pytest.raises(NotImplementedError):
            r(torch.randn(2, 8))