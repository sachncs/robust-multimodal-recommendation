"""Tests for morel.core.device."""

from __future__ import annotations

import pytest
import torch

from morel.core.device import Device, device, to


def test_device_string_passthrough() -> None:
    d = device("cpu")
    assert d.type == "cpu"


def test_device_falls_back_to_cpu_when_cuda_unavailable() -> None:
    if torch.cuda.is_available():
        return
    assert device("cuda").type == "cpu"


def test_device_none_resolves() -> None:
    d = device(None)
    assert isinstance(d, torch.device)


def test_device_torch_object() -> None:
    d = device(torch.device("cpu"))
    assert d.type == "cpu"


def test_device_enum_values() -> None:
    assert Device.CPU.value == "cpu"
    assert Device.CUDA.value == "cuda"


def test_to_moves_tensor() -> None:
    t = torch.zeros(2)
    moved = to(t, "cpu")
    assert moved.device.type == "cpu"


def test_auto_is_accepted_because_it_is_the_config_default() -> None:
    """Regression: device("auto") raised, yet Config.device defaults to "auto"."""
    from morel.core.config import Config

    assert Config().device == "auto"
    assert device("auto") == device(None)


@pytest.mark.parametrize("value", ["auto", "AUTO", " auto ", "", "default"])
def test_auto_like_values_resolve_to_the_detected_device(value: str) -> None:
    assert device(value) == device(None)


def test_explicit_device_object_passes_through() -> None:
    assert device(torch.device("cpu")) == torch.device("cpu")
