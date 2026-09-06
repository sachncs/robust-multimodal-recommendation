"""Typed exception hierarchy for morel.

Every error raised by the library is a `Error`. Specializations live in
submodules and re-export here. The base class itself ends in `Error`
(see Rule A); subclasses are named after the domain concept (see
Rule D single-word naming) rather than redundantly appending `Error`.
"""

from __future__ import annotations


class Error(Exception):
    """Base class for every exception raised by morel."""


class Datum(Error):  # noqa: N818  # Rule D: single-word class name (domain concept).
    """Data acquisition, validation, or loading failures."""


class Cfg(Error):  # noqa: N818  # Rule D: single-word class name (domain concept).
    """Invalid or inconsistent configuration."""


class Model(Error):  # noqa: N818  # Rule D: single-word class name (domain concept).
    """Model construction, forward, or parameter validation failures."""


class Net(Error):  # noqa: N818  # Rule D: single-word class name (domain concept).
    """Graph construction, invariant violation, or retrieval failures."""


class Train(Error):  # noqa: N818  # Rule D: single-word class name (domain concept).
    """Training loop failures (NaN loss, missing checkpoint, etc.)."""


class Rate(Error):  # noqa: N818  # Rule D: single-word class name (domain concept).
    """Evaluation failures (empty score matrix, etc.)."""


class Shape(Error):  # noqa: N818  # Rule D: single-word class name (domain concept).
    """Tensor shape mismatch."""


class Determinism(Error):  # noqa: N818  # Rule D: single-word class name (domain concept).
    """Reproducibility invariant violated."""


__all__ = [
    "Cfg",
    "Datum",
    "Determinism",
    "Error",
    "Model",
    "Net",
    "Rate",
    "Shape",
    "Train",
]
