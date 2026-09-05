"""Core protocols and dataclasses shared across morel.

Every domain module imports its primitives from here. This module imports nothing
from morel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class Modality(Protocol):
    """One modality in the system."""

    name: str
    dim: int
    dtype: torch.dtype


@runtime_checkable
class Mask(Protocol):
    """Modality availability mask.

    Semantics: 1 = observed, 0 = missing. Shape ``(items, modalities)``.
    """

    @property
    def data(self) -> np.ndarray:
        """Return the underlying binary array."""

    @property
    def items(self) -> int:
        """Return the number of items."""

    @property
    def modalities(self) -> int:
        """Return the number of modalities."""

    def kept(self, item: int) -> np.ndarray:
        """Return which modalities are kept for a given item."""

    def missing(self, item: int) -> np.ndarray:
        """Return which modalities are missing for a given item."""


@runtime_checkable
class Graph(Protocol):
    """Sparse graph abstraction."""

    @property
    def nodes(self) -> int:
        """Return the number of nodes."""

    @property
    def edges(self) -> int:
        """Return the number of edges."""

    def adjacency(self) -> Any:
        """Return the underlying sparse adjacency (scipy CSR preferred)."""


@dataclass(frozen=True)
class Embedding:
    """Tensor wrapper that exposes shape, dtype, device explicitly.

    Carries a single torch tensor with its semantic name. Used at module
    boundaries to make data flow auditable.
    """

    name: str
    tensor: torch.Tensor

    @property
    def shape(self) -> torch.Size:
        """Return the shape of the underlying tensor."""
        return self.tensor.shape

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the underlying tensor."""
        return self.tensor.dtype

    @property
    def device(self) -> torch.device:
        """Return the device of the underlying tensor."""
        return self.tensor.device

    @property
    def requires_grad(self) -> bool:
        """Return whether the underlying tensor requires gradients."""
        return self.tensor.requires_grad

    def to(self, device: torch.device | str) -> "Embedding":
        """Return a new Embedding on the given device."""
        return Embedding(name=self.name, tensor=self.tensor.to(device))


__all__ = ["Modality", "Mask", "Graph", "Embedding"]
