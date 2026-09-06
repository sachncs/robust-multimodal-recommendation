"""Atomic save/load for data artifacts with manifest verification."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from morel.core.errors import DataError
from morel.data import manifest


def atomic_write(target: Path, writer: Callable[[Path], None]) -> Path:
    """Write to a sibling tempfile, then atomically replace the target.

    The tempfile name is constructed so that ``np.savez`` and ``sp.save_npz``
    do not silently append a ``.npz`` suffix. Both libraries append ``.npz``
    when the path does not already end in that suffix; we therefore keep the
    target's suffix in the tempfile name.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix or ".tmp"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix=target.stem + ".", dir=target.parent)
    os.close(fd)
    try:
        writer(Path(tmp_path))
        os.replace(tmp_path, target)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return target


def save_npz(
    target: Path | str,
    *,
    manifest_obj: manifest.Manifest | None = None,
    **arrays: np.ndarray,
) -> Path:
    """Atomically save one or more numpy arrays plus an optional manifest.

    Args:
        target: Destination ``.npz`` path.
        manifest_obj: Optional manifest to save as a sidecar.
        **arrays: Named arrays to include.

    Returns
    -------
        The destination path.
    """
    if not arrays:
        raise DataError("save_npz requires at least one array")
    final = Path(target).resolve()
    # A key literally named ``allow_pickle`` would bind to savez's own keyword;
    # that raises rather than silently dropping data, so the narrowing is safe.
    atomic_write(final, lambda tmp: np.savez(tmp, **arrays))  # type: ignore[arg-type]
    if manifest_obj is not None:
        manifest.save(final, manifest_obj)
    return final


def load_npz(
    target: Path | str, *, expected_config_hash: str | None = None
) -> dict[str, np.ndarray]:
    """Load a ``.npz`` artifact and verify the manifest if present.

    Args:
        target: Path to the ``.npz`` file.
        expected_config_hash: If set, raise on manifest mismatch.

    Returns
    -------
        Dict mapping array names to numpy arrays.
    """
    path = Path(target)
    if not path.exists():
        raise DataError(f"npz artifact not found: {path}")
    if manifest.path_for(path).exists():
        manifest.load(path, expected_config_hash=expected_config_hash)
    with np.load(path, allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def save_graph(
    target: Path | str,
    graph: sp.spmatrix,
    *,
    manifest_obj: manifest.Manifest | None = None,
) -> Path:
    """Atomically save a sparse graph with manifest.

    Stores data, indices, indptr, and shape as a regular npz so that the
    file can be loaded without scipy's filename-suffix shenanigans.
    """
    final = Path(target).resolve()
    coo = sp.coo_matrix(graph)

    def writer(tmp: Path) -> None:
        np.savez(
            tmp,
            data=coo.data.astype(np.float32, copy=False),
            row=coo.row.astype(np.int64, copy=False),
            col=coo.col.astype(np.int64, copy=False),
            shape=np.asarray(coo.shape, dtype=np.int64),
        )

    atomic_write(final, writer)
    if manifest_obj is not None:
        manifest.save(final, manifest_obj)
    return final


def load_graph(target: Path | str, *, expected_config_hash: str | None = None) -> sp.spmatrix:
    """Load a sparse graph and verify its manifest."""
    path = Path(target)
    if not path.exists():
        raise DataError(f"graph artifact not found: {path}")
    if manifest.path_for(path).exists():
        manifest.load(path, expected_config_hash=expected_config_hash)
    with np.load(path, allow_pickle=False) as npz:
        data = npz["data"]
        row = npz["row"]
        col = npz["col"]
        shape = tuple(int(s) for s in npz["shape"])
    coo = sp.coo_matrix((data, (row, col)), shape=shape)
    return coo.tocsr()


__all__ = ["load_graph", "load_npz", "save_graph", "save_npz"]
