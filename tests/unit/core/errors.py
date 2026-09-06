"""Tests for morel.core.errors."""

from __future__ import annotations

import pytest

from morel.core.errors import (
    ConfigError,
    DataError,
    DeterminismError,
    EvalError,
    GraphError,
    ModelError,
    MorelError,
    ShapeError,
    TrainError,
)


class Checker:
    """Aggregated test methods for this module."""

    def specializations(self, cls: type) -> None:
        assert issubclass(cls, MorelError)

    def can(self) -> None:
        with pytest.raises(MorelError):
            raise GraphError("boom")