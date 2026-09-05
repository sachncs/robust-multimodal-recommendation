"""Tests for morel.codebook."""

from __future__ import annotations

import torch

from morel.codebook import GumbelVQ, VQ, balance, usage
from morel.route import Top


def test_vq_quantizes_to_nearest() -> None:
    vq = VQ(dim=4, size=8)
    z, idx = vq(torch.randn(3, 4))
    assert z.shape == (3, 4)
    assert idx.shape == (3,)


def test_vq_invalid_size() -> None:
    import pytest

    with pytest.raises(ValueError):
        VQ(dim=4, size=0)


def test_gumbel_vq_with_top_router() -> None:
    router = Top(dim=8, k=10, p=3, tau=0.5)
    gvq = GumbelVQ(dim=8, size=10, router=router)
    q, p = gvq(torch.randn(2, 8), training=True)
    assert q.shape == (2, 8)
    assert p.shape == (2, 10)


def test_usage_loss_zero_at_uniform() -> None:
    probs = torch.full((16, 10), 0.1)
    assert float(usage(probs)) < 1e-5


def test_balance_loss_at_uniform() -> None:
    probs = torch.full((16, 10), 0.1)
    # K * sum(bar_p^2) = 10 * (10 * 0.01) = 1.0
    assert abs(float(balance(probs)) - 1.0) < 1e-3


def test_balance_loss_positive() -> None:
    probs = torch.softmax(torch.randn(16, 10), dim=-1)
    assert float(balance(probs)) > 0


def test_usage_loss_positive_for_imbalanced() -> None:
    probs = torch.zeros(16, 10)
    probs[:, 0] = 1.0
    assert float(usage(probs)) > 1.0
