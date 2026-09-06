"""Tests for morel.codebook."""

from __future__ import annotations

import torch

from morel.codebook import (
    VQ,
    Codebook,
    GumbelVQ,
    IdentityCodebook,
    balance,
    usage,
)
from morel.route import Top


class Checker:
    """Aggregated test methods for this module."""

    def vq(self) -> None:
        vq = VQ(dim=4, size=8)
        z, one_hot = vq(torch.randn(3, 4))
        assert z.shape == (3, 4)
        # VQ returns a one-hot routing vector so its output shape matches GumbelVQ.
        assert one_hot.shape == (3, 8)
        assert torch.allclose(one_hot.sum(dim=-1), torch.ones(3))

    def invalid(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="size must be positive"):
            VQ(dim=4, size=0)

    def gumbel(self) -> None:
        router = Top(dim=8, k=10, p=3, tau=0.5)
        gvq = GumbelVQ(dim=8, size=10, router=router)
        q, p = gvq(torch.randn(2, 8), training=True)
        assert q.shape == (2, 8)
        assert p.shape == (2, 10)

    def a(self) -> None:
        from morel.route import Dense

        gvq = GumbelVQ(dim=4, size=10, router=Dense(dim=4, k=10))
        assert isinstance(gvq, Codebook)

    def identity(self) -> None:
        cb = IdentityCodebook(dim=4, size=8)
        x = torch.randn(2, 4)
        out, probs = cb(x, training=False)
        assert torch.equal(out, x)
        assert probs.shape == (2, 8)
        assert torch.allclose(probs, torch.full((2, 8), 1.0 / 8))

    def codebook(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="size must be positive"):
            IdentityCodebook(dim=4, size=0)

    def usage(self) -> None:
        probs = torch.full((16, 10), 0.1)
        assert float(usage(probs)) < 1e-5

    def balance(self) -> None:
        probs = torch.full((16, 10), 0.1)
        # K * sum(bar_p^2) = 10 * (10 * 0.01) = 1.0
        assert abs(float(balance(probs)) - 1.0) < 1e-3

    def loss(self) -> None:
        probs = torch.softmax(torch.randn(16, 10), dim=-1)
        assert float(balance(probs)) > 0

    def positive(self) -> None:
        probs = torch.zeros(16, 10)
        probs[:, 0] = 1.0
        assert float(usage(probs)) > 1.0

    def seed(self) -> None:
        torch.manual_seed(1)
        first = VQ(dim=8, size=16, seed=5)
        torch.manual_seed(9999)
        second = VQ(dim=8, size=16, seed=5)
        assert torch.equal(first.embeddings.weight, second.embeddings.weight)

    def makes(self) -> None:
        torch.manual_seed(1)
        first = GumbelVQ(dim=8, size=16, router=Top(dim=8, k=16, p=4), seed=5)
        torch.manual_seed(9999)
        second = GumbelVQ(dim=8, size=16, router=Top(dim=8, k=16, p=4), seed=5)
        assert torch.equal(first.codebook.weight, second.codebook.weight)