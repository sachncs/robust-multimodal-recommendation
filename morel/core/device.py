"""Centralized device handling.

Every `tensor.to(...)` call in the codebase should ultimately route through
`device()` or `to()`. Modules never hardcode ``cuda``.
"""

from __future__ import annotations

from enum import Enum

import torch


class Device(str, Enum):
    """Supported device types."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


def device(prefer: str | torch.device | None = None) -> torch.device:
    """Resolve a torch device.

    Args:
        prefer: Explicit device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``,
            ``"mps"``) or torch.device. ``None``, ``"auto"`` or an empty
            string select CUDA if available, else MPS if available, else CPU.
            ``"auto"`` is accepted because it is the default of
            ``Config.device``, and a config default that its own resolver
            rejects would be a trap.

    Returns
    -------
        The resolved ``torch.device``.
    """
    if isinstance(prefer, torch.device):
        return prefer
    if prefer is None or str(prefer).strip().lower() in {"", "auto", "default"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    text = str(prefer).strip().lower()
    if text == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(text)


def to(tensor: torch.Tensor, target: torch.device | str) -> torch.Tensor:
    """Move a tensor to the target device.

    Args:
        tensor: Input tensor.
        target: Target device.

    Returns
    -------
        Tensor on the target device.
    """
    return tensor.to(device(target))


__all__ = ["Device", "device", "to"]
