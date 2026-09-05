"""Typed exception hierarchy for morel.

Every error raised by the library is a `MorelError`. Specializations live in
submodules and re-export here.
"""

from __future__ import annotations


class MorelError(Exception):
    """Base class for every exception raised by morel."""


class Data(MorelError):
    """Data acquisition, validation, or loading failures."""


class Config(MorelError):
    """Invalid or inconsistent configuration."""


class Model(MorelError):
    """Model construction, forward, or parameter validation failures."""


class Graph(MorelError):
    """Graph construction, invariant violation, or retrieval failures."""


class Train(MorelError):
    """Training loop failures (NaN loss, missing checkpoint, etc.)."""


class Eval(MorelError):
    """Evaluation failures (empty score matrix, etc.)."""


class Shape(MorelError):
    """Tensor shape mismatch."""


class Determinism(MorelError):
    """Reproducibility invariant violated."""


__all__ = [
    "MorelError",
    "Data",
    "Config",
    "Model",
    "Graph",
    "Train",
    "Eval",
    "Shape",
    "Determinism",
]
