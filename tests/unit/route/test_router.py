"""Tests for morel.route."""

from __future__ import annotations

import pytest
import torch

from morel.route import Dense, Fixed, Gumbel, Router, Top, Weights, build


def test_dense_routing() -> None:
    r = Dense(dim=8, k=5, tau=1.0)
    out = r(torch.randn(2, 8))
    assert out.probs.shape == (2, 5)
    assert torch.allclose(out.probs.sum(-1), torch.ones(2), atol=1e-5)


def test_top_routing_sums_to_one() -> None:
    r = Top(dim=8, k=10, p=3, tau=0.5)
    out = r(torch.randn(4, 8), training=True)
    assert out.probs.shape == (4, 10)
    assert torch.allclose(out.probs.sum(-1), torch.ones(4), atol=1e-5)


def test_top_routing_is_sparse() -> None:
    r = Top(dim=8, k=10, p=3, tau=0.5)
    out = r(torch.randn(4, 8), training=False)
    nonzero = (out.probs > 0).sum(-1)
    assert (nonzero == 3).all()


def test_top_p_greater_than_k_raises() -> None:
    with pytest.raises(ValueError):
        Top(dim=8, k=5, p=10)


def test_top_invalid_tau() -> None:
    with pytest.raises(ValueError):
        Top(dim=8, k=5, p=2, tau=0.0)


def test_gumbel_routing_sums_to_one() -> None:
    r = Gumbel(dim=8, k=10, tau=0.5)
    out = r(torch.randn(4, 8), training=True)
    assert torch.allclose(out.probs.sum(-1), torch.ones(4), atol=1e-5)


def test_build_factory() -> None:
    r = build("top", dim=8, k=10, p=3, tau=0.5)
    assert isinstance(r, Top)
    r = build("dense", dim=8, k=10, p=3, tau=0.5)
    assert isinstance(r, Dense)


def test_build_unknown() -> None:
    with pytest.raises(ValueError):
        build("nope", dim=8, k=10, p=3, tau=0.5)


def test_weights_dataclass_shape() -> None:
    w = Weights(probs=torch.zeros(2, 5))
    assert w.shape == (2, 5)


def test_fixed_router_raises_without_index() -> None:
    r = Fixed(k=5)
    with pytest.raises(NotImplementedError):
        r(torch.randn(2, 8))
