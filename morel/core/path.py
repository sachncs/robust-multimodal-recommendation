"""Path resolution for morel artifacts.

Single source of truth for where raw, processed, checkpoint, and run artifacts
live. Honors the ``MOREL_DATA_DIR`` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

from morel.core.errors import ConfigError


def root() -> Path:
    """Return the morel data root directory.

    Defaults to ``./data`` relative to the current working directory.
    Override with ``MOREL_DATA_DIR``.
    """
    return Path(os.environ.get("MOREL_DATA_DIR", "data")).resolve()


def raw() -> Path:
    """Return the raw data directory."""
    return root() / "raw"


def processed(name: str) -> Path:
    """Return the processed data directory for a given dataset name.

    Args:
        name: Dataset identifier.

    Returns
    -------
        Path under ``<root>/processed/<name>``.
    """
    if not name or not name.replace("_", "").replace("-", "").isalnum():
        raise ConfigError(f"invalid dataset name: {name!r}")
    return root() / "processed" / name


def features(name: str) -> Path:
    """Return the features directory for a dataset."""
    return processed(name) / "features"


def graphs(name: str) -> Path:
    """Return the graphs directory for a dataset."""
    return processed(name) / "graphs"


def checkpoints(run: str) -> Path:
    """Return the checkpoints directory for a run.

    Args:
        run: Run identifier.

    Returns
    -------
        Path under ``<root>/checkpoints/<run>``.
    """
    if not run:
        raise ConfigError("run identifier must be non-empty")
    return root() / "checkpoints" / run


def runs(run: str) -> Path:
    """Return the run-artifact directory."""
    if not run:
        raise ConfigError("run identifier must be non-empty")
    return root() / "runs" / run


def manifest(artifact: str) -> Path:
    """Return the manifest sidecar path for an artifact."""
    return Path(artifact).with_suffix(Path(artifact).suffix + ".manifest.json")


__all__ = ["root", "raw", "processed", "features", "graphs", "checkpoints", "runs", "manifest"]
