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


@pytest.mark.parametrize(
    "cls",
    [
        DataError,
        ConfigError,
        ModelError,
        GraphError,
        TrainError,
        EvalError,
        ShapeError,
        DeterminismError,
    ],
)
def test_specializations_extend_base(cls: type) -> None:
    assert issubclass(cls, MorelError)


def test_can_raise_and_catch() -> None:
    with pytest.raises(MorelError):
        raise GraphError("boom")
