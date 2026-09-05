"""Validation primitives for data inputs."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from morel.core.errors import DataError, ShapeError


def interactions(
    user: np.ndarray, item: np.ndarray, users: int, items: int
) -> None:
    """Validate a user-item interaction pair array.

    Args:
        user: 1-D array of user indices.
        item: 1-D array of item indices, same length as ``user``.
        users: Expected total number of users.
        items: Expected total number of items.

    Raises:
        DataError: On shape, dtype, or range violations.
    """
    if user.ndim != 1 or item.ndim != 1:
        raise ShapeError("user and item must be 1-D")
    if user.shape != item.shape:
        raise ShapeError(f"shape mismatch: user {user.shape} vs item {item.shape}")
    if user.size == 0:
        raise DataError("interactions are empty")
    if user.min() < 0 or item.min() < 0:
        raise DataError("negative indices not allowed")
    if user.max() >= users:
        raise DataError(f"user index {int(user.max())} >= users ({users})")
    if item.max() >= items:
        raise DataError(f"item index {int(item.max())} >= items ({items})")


def features(payload: dict[str, np.ndarray], *, items: int) -> None:
    """Validate per-modality feature arrays.

    Args:
        payload: Dict mapping modality name to ``(items, dim)`` float array.
        items: Expected number of items.

    Raises:
        DataError: On shape, dtype, or value-range violations.
    """
    if not payload:
        raise DataError("features dict is empty")
    for name, array in payload.items():
        if array.ndim != 2:
            raise ShapeError(f"feature {name!r} must be 2-D, got {array.ndim}-D")
        if array.shape[0] != items:
            raise ShapeError(
                f"feature {name!r} row count {array.shape[0]} != items ({items})"
            )
        if not np.isfinite(array).all():
            raise DataError(f"feature {name!r} contains NaN or Inf")
        if array.dtype != np.float32:
            raise DataError(
                f"feature {name!r} dtype {array.dtype} != float32"
            )


def graph(adj: sp.spmatrix) -> None:
    """Validate a sparse graph adjacency.

    Args:
        adj: A scipy sparse matrix.

    Raises:
        DataError: On invariant violations.
    """
    if adj.ndim != 2:
        raise ShapeError(f"graph must be 2-D, got {adj.ndim}-D")
    rows, cols = adj.shape
    if rows != cols:
        raise DataError(f"graph must be square, got {adj.shape}")
    diag = adj.diagonal()
    if (diag != 0).any():
        raise DataError("graph has self-loops")
    coo = adj.tocoo()
    if (coo.data < 0).any():
        raise DataError("graph has negative edge weights")


def mask(mask: np.ndarray) -> None:
    """Validate a modality availability mask.

    Args:
        mask: 2-D binary array of shape ``(items, modalities)``.

    Raises:
        DataError: On shape, dtype, or value-range violations.
    """
    if mask.ndim != 2:
        raise ShapeError(f"mask must be 2-D, got {mask.ndim}-D")
    if mask.shape[0] == 0:
        raise DataError("mask has no items")
    unique = np.unique(mask)
    if not np.all(np.isin(unique, [0.0, 1.0])):
        raise DataError(f"mask values must be 0 or 1, got {unique}")
    rowsums = mask.sum(axis=1)
    if (rowsums == 0).any():
        raise DataError("mask has rows with no kept modalities")


__all__ = ["interactions", "features", "graph", "mask"]
