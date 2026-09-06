"""Tests for morel.complete."""

from __future__ import annotations

import pytest
import torch

from morel.complete import Decoders


class Checker:
    """Aggregated test methods for this module."""

    def decoders() -> None:
        d = Decoders(latent_dim=8, dims={"v": 4, "t": 2})
        z = torch.randn(3, 8)
        out = d(z)
        assert out["v"].shape == (3, 4)
        assert out["t"].shape == (3, 2)

    def mask() -> None:
        d = Decoders(latent_dim=8, dims={"v": 4})
        z = torch.randn(3, 8)
        mask = torch.tensor([[0.0], [1.0], [0.0]])
        out = d(z, mask)
        assert out["v"].shape == (3, 4)

    def custom() -> None:
        d = Decoders(latent_dim=8, dims={"v": 4}, hidden=16)
        z = torch.randn(2, 8)
        out = d(z)
        assert out["v"].shape == (2, 4)

    def invalid() -> None:
        with pytest.raises(ValueError, match="latent_dim must be positive"):
            Decoders(latent_dim=0, dims={"v": 4})

    def dims() -> None:
        with pytest.raises(ValueError, match="dims must not be empty"):
            Decoders(latent_dim=8, dims={})
