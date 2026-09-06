"""Deterministic seeding for the entire runtime.

`seed(seed)` makes every supported library deterministic for the current run.
`state()` and `restore()` snapshot/restore the full RNG state for resume.

Covers: torch, torch.cuda, numpy, random, PYTHONHASHSEED, cudnn.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor


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
    if not isinstance(value, int):
        raise TypeError(f"seed must be int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"seed must be non-negative, got {value}")
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

    The field types are deliberately loose: each library returns its own
    opaque state blob (``numpy`` returns a 5-tuple for the legacy generator
    and a dict for the newer one), and morel only ever passes them straight
    back to the library that produced them.
    """

    python: tuple[Any, ...]
    numpy: dict[str, Any] | tuple[Any, ...]
    # Annotated via the direct ``Tensor`` import because the field name
    # ``torch`` shadows the module inside this class body.
    torch: Tensor
    cuda: list[Tensor] | None


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


@contextmanager
def deterministic(value: int) -> Iterator[None]:
    """Seed every RNG for the duration of the block, then restore prior state.

    Use this to make a bounded region of work — typically parameter
    initialization — reproducible without leaking a global reseed into the
    surrounding program. On exit the RNG state of ``random``, ``numpy`` and
    ``torch`` is restored exactly as it was on entry, so callers that draw
    their own random numbers are unaffected by the block.

    This is the mechanism that lets a model constructor be deterministic as
    a *property of the object* rather than a property of whatever the caller
    happened to do beforehand.

    Args:
        value: Non-negative integer seed.

    Yields
    ------
        ``None``. The block body runs under the given seed.

    Examples
    --------
        >>> import torch
        >>> from morel.core.seed import deterministic
        >>> with deterministic(0):
        ...     a = torch.rand(3)
        >>> with deterministic(0):
        ...     b = torch.rand(3)
        >>> bool(torch.equal(a, b))
        True
    """
    snapshot = state()
    try:
        seed(value)
        yield
    finally:
        restore(snapshot)


__all__ = ["State", "deterministic", "restore", "seed", "state"]
