"""Typed exception hierarchy for morel.

Every error raised by the library is a `Error`. Specializations live in
submodules and re-export here.
"""

from __future__ import annotations


class Error(Exception):
    """Base class for every exception raised by morel."""


class DataError(Error):
    """Data acquisition, validation, or loading failures."""


class Cfg(Error):
    """Invalid or inconsistent configuration."""


class Model(Error):
    """Model construction, forward, or parameter validation failures."""


class GraphError(Error):
    """Graph construction, invariant violation, or retrieval failures."""


class Train(Error):
    """Training loop failures (NaN loss, missing checkpoint, etc.)."""


class EvalError(Error):
    """Evaluation failures (empty score matrix, etc.)."""


class Shape(Error):
    """Tensor shape mismatch."""


class Determinism(Error):
    """Reproducibility invariant violated."""


__all__ = [
    "Cfg",
    "DataError",
    "Determinism",
    "EvalError",
    "GraphError",
    "Model",
    "Error",
    "Shape",
    "Train",
]
