"""Distributed runtime primitives.

Single entry point for distributed training. ``init()`` reads
``MASTER_ADDR``/``MASTER_PORT``/``RANK``/``WORLD_SIZE``/``LOCAL_RANK`` env vars
(the standard ``torchrun`` contract) and selects the right backend
(NCCL on CUDA, Gloo otherwise). ``init()`` is idempotent.

All helpers degrade to single-process semantics when ``WORLD_SIZE`` is
unset or equals 1, so non-distributed callers do not need to branch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from morel.core.errors import MorelError


@dataclass
class DistributedState:
    """Module-level distributed runtime state."""

    backend: str | None = None
    initialized: bool = False


state = DistributedState()


def init(backend: str | None = None) -> dict[str, Any]:
    """Initialize the default process group.

    Args
    ----
    backend : str | None
        Explicit backend choice. Defaults to ``nccl`` on CUDA, ``gloo``
        on CPU.

    Returns
    -------
    dict[str, Any]
        Information about the initialised process group.

    Raises
    ------
    MorelError
        If torch.distributed fails to initialise.
    """
    if state.initialized:
        return {
            "rank": rank(),
            "world_size": world_size(),
            "local_rank": local_rank(),
            "backend": state.backend,
        }
    if not torch.distributed.is_available():
        state.backend = None
        state.initialized = True
        return {
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "backend": None,
        }
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size_env <= 1:
        state.backend = None
        state.initialized = True
        return {
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "backend": None,
        }
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    try:
        torch.distributed.init_process_group(backend=backend)
    except Exception as exc:
        raise MorelError(f"failed to init distributed group with backend={backend}: {exc}") from exc
    state.backend = backend
    state.initialized = True
    if torch.cuda.is_available() and local_rank() < torch.cuda.device_count():
        torch.cuda.set_device(local_rank())
    return {
        "rank": rank(),
        "world_size": world_size(),
        "local_rank": local_rank(),
        "backend": backend,
    }


def is_initialized() -> bool:
    """Return whether the runtime has been initialised."""
    return state.initialized


def rank() -> int:
    """Return the global rank of the current process (0 when single-process)."""
    if not state.initialized:
        return 0
    if not torch.distributed.is_available():
        return 0
    if not torch.distributed.is_initialized():
        return 0
    return int(torch.distributed.get_rank())


def world_size() -> int:
    """Return the world size (1 when single-process)."""
    if not state.initialized:
        return 1
    if not torch.distributed.is_available():
        return 1
    if not torch.distributed.is_initialized():
        return 1
    return int(torch.distributed.get_world_size())


def local_rank() -> int:
    """Return the local rank on the current node."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_rank_zero() -> bool:
    """Return True for the rank-zero process (always True when single-process)."""
    return rank() == 0


def barrier() -> None:
    """Block until all ranks reach this point."""
    if not state.initialized:
        return
    if not torch.distributed.is_available():
        return
    if not torch.distributed.is_initialized():
        return
    if world_size() <= 1:
        return
    torch.distributed.barrier()


def reduce_mean(value: float | torch.Tensor) -> float:
    """All-reduce a scalar across ranks and return its mean."""
    if not state.initialized or world_size() <= 1:
        return float(value)
    tensor = torch.as_tensor(value, dtype=torch.float64)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor /= world_size()
    return float(tensor.item())


def cleanup() -> None:
    """Destroy the default process group if initialised."""
    if (
        state.initialized
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.destroy_process_group()
    state.initialized = False
    state.backend = None


__all__ = [
    "DistributedState",
    "init",
    "is_initialized",
    "rank",
    "world_size",
    "local_rank",
    "is_rank_zero",
    "barrier",
    "reduce_mean",
    "cleanup",
    "state",
]
