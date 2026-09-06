"""Typed exception hierarchy for morel.

Every error raised by the library is a `MorelError`. Specializations live in
submodules and re-export here.
"""

from __future__ import annotations


class MorelError(Exception):
    """Base class for every exception raised by morel."""


class DataError(MorelError):
    """Data acquisition, validation, or loading failures."""


class ConfigError(MorelError):
    """Invalid or inconsistent configuration."""


class ModelError(MorelError):
    """Model construction, forward, or parameter validation failures."""


class GraphError(MorelError):
    """Graph construction, invariant violation, or retrieval failures."""


class TrainError(MorelError):
    """Training loop failures (NaN loss, missing checkpoint, etc.)."""


class EvalError(MorelError):
    """Evaluation failures (empty score matrix, etc.)."""


class ShapeError(MorelError):
    """Tensor shape mismatch."""


class DeterminismError(MorelError):
    """Reproducibility invariant violated."""


__all__ = [
    "MorelError",
    "DataError",
    "ConfigError",
    "ModelError",
    "GraphError",
    "TrainError",
    "EvalError",
    "ShapeError",
    "DeterminismError",
]
