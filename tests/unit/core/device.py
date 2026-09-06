"""Tests for morel.core.device."""

from __future__ import annotations

import pytest
import torch

from morel.core.device import Device, device, to


class Checker:
    """Aggregated test methods for this module."""

    def device(self) -> None:
        d = device("cpu")
        assert d.type == "cpu"

    def falls(self) -> None:
        if torch.cuda.is_available():
            return
        assert device("cuda").type == "cpu"

    def none(self) -> None:
        d = device(None)
        assert isinstance(d, torch.device)

    def torch(self) -> None:
        d = device(torch.device("cpu"))
        assert d.type == "cpu"

    def enum(self) -> None:
        assert Device.CPU.value == "cpu"
        assert Device.CUDA.value == "cuda"

    def to(self) -> None:
        t = torch.zeros(2)
        moved = to(t, "cpu")
        assert moved.device.type == "cpu"

    def auto(self) -> None:
        """Regression: device("auto") raised, yet Config.device defaults to "auto"."""
        from morel.core.config import Config

        assert Config().device == "auto"
        assert device("auto") == device(None)

    @pytest.mark.parametrize("value", ["auto", "AUTO", " auto ", "", "default"])
    def like(self, value: str) -> None:
        assert device(value) == device(None)

    def explicit(self) -> None:
        assert device(torch.device("cpu")) == torch.device("cpu")