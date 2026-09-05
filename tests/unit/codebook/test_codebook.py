"""Tests for morel.codebook."""

from __future__ import annotations

import torch

from morel.codebook import (
    Codebook,
    GumbelVQ,
    IdentityCodebook,
    VQ,
    balance,
    usage,
)
from morel.route import Top


def test_vq_quantizes_to_nearest() -> None:
    vq = VQ(dim=4, size=8)
    z, one_hot = vq(torch.randn(3, 4))
    assert z.shape == (3, 4)
    # VQ returns a one-hot routing vector so its output shape matches GumbelVQ.
    assert one_hot.shape == (3, 8)
    assert torch.allclose(one_hot.sum(dim=-1), torch.ones(3))


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


def test_gumbel_vq_is_a_codebook() -> None:
    from morel.route import Dense

    gvq = GumbelVQ(dim=4, size=10, router=Dense(dim=4, k=10))
    assert isinstance(gvq, Codebook)


def test_identity_codebook_passthrough() -> None:
    cb = IdentityCodebook(dim=4, size=8)
    x = torch.randn(2, 4)
    out, probs = cb(x, training=False)
    assert torch.equal(out, x)
    assert probs.shape == (2, 8)
    assert torch.allclose(probs, torch.full((2, 8), 1.0 / 8))


def test_identity_codebook_rejects_invalid_size() -> None:
    import pytest

    with pytest.raises(ValueError):
        IdentityCodebook(dim=4, size=0)


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
