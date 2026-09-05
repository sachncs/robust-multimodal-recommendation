"""Tests for morel.core.device."""

from __future__ import annotations

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
