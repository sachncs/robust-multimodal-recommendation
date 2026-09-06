"""Deterministic seeding for the entire runtime.

`seed(seed)` makes every supported library deterministic for the current run.
`state()` and `restore()` snapshot/restore the full RNG state for resume.

Covers: torch, torch.cuda, numpy, random, PYTHONHASHSEED, cudnn.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def seed(value: int) -> int:
    """Seed every RNG in the runtime. Returns the seed used.

    Sets:
        - ``torch.manual_seed``
        - ``torch.cuda.manual_seed_all`` if CUDA is available
        - ``numpy.random.seed``
        - ``random.seed``
        - ``PYTHONHASHSEED`` (must be set before Python starts for full effect;
          this is a best-effort runtime setting)
        - ``cudnn.deterministic = True``
        - ``cudnn.benchmark = False``

    Args:
        value: Non-negative integer seed.

    Returns
    -------
        The seed used.

    Raises
    ------
        ValueError: If the seed is negative.
    """
    if value < 0:
        raise ValueError(f"seed must be non-negative, got {value}")
    if not isinstance(value, int):
        raise TypeError(f"seed must be int, got {type(value).__name__}")
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    os.environ["PYTHONHASHSEED"] = str(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return value


@dataclass
class State:
    """Snapshot of all RNG state in the runtime.

    Restore with `restore(state)` for exact reproducibility across process
    restarts.
    """

    python: tuple[Any, ...]
    numpy: dict[str, Any]
    torch: torch.Tensor
    cuda: list[torch.Tensor] | None


def state() -> State:
    """Snapshot the current RNG state of every library."""
    torch_state = torch.get_rng_state()
    cuda_state: list[torch.Tensor] | None = None
    if torch.cuda.is_available():
        cuda_state = [
            torch.cuda.get_rng_state(device) for device in range(torch.cuda.device_count())
        ]
    np_state = np.random.get_state()
    return State(
        python=random.getstate(),
        numpy=np_state,
        torch=torch_state,
        cuda=cuda_state,
    )


def restore(snapshot: State) -> None:
    """Restore the RNG state from a snapshot."""
    random.setstate(snapshot.python)
    np.random.set_state(snapshot.numpy)
    torch.set_rng_state(snapshot.torch)
    if snapshot.cuda is not None and torch.cuda.is_available():
        for device, tensor in enumerate(snapshot.cuda):
            torch.cuda.set_rng_state(tensor, device=device)


__all__ = ["State", "restore", "seed", "state"]
