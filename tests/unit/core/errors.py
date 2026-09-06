"""Tests for morel.core.errors."""

from __future__ import annotations

import pytest

from morel.core.errors import (
    Cfg,
    DataError,
    Determinism,
    EvalError,
    GraphError,
    Model,
    Error,
    Shape,
    Train,
)


class Checker:
    """Aggregated test methods for this module."""

    @pytest.mark.parametrize(
        "cls",
        [
            DataError,
            Cfg,
            Model,
            GraphError,
            Train,
            EvalError,
            Shape,
            Determinism,
        ],
    )
    def specializations(self, cls: type) -> None:
        assert issubclass(cls, Error)

    def can(self) -> None:
        with pytest.raises(Error):
            raise GraphError("boom")