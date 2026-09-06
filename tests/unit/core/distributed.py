"""Tests for morel.core.distributed."""

from __future__ import annotations

import pytest

from morel.core import distributed as dist


@pytest.fixture(autouse=True)
def reset():
    """Reset module-level distributed state before each test."""
    dist.state.initialized = False
    dist.state.backend = None
    yield
    dist.state.initialized = False
    dist.state.backend = None


class Checker:
    """Aggregated test methods for this module."""

    def default(self) -> None:
        assert dist.world_size() == 1

    def rank(self) -> None:
        assert dist.rank() == 0

    def zero(self) -> None:
        assert dist.is_lead() is True

    def init(self) -> None:
        info = dist.init()
        assert info["rank"] == 0
        assert info["world_size"] == 1
        assert dist.initialized() is True

    def idempotent(self) -> None:
        dist.init()
        dist.init()
        assert dist.initialized() is True
        assert dist.rank() == 0

    def barrier(self) -> None:
        dist.barrier()
        assert dist.is_lead() is True

    def reduce(self) -> None:
        dist.init()
        assert dist.mean(7.5) == 7.5

    def local(self) -> None:
        import os

        os.environ["LOCAL_RANK"] = "3"
        try:
            assert dist.local_rank() == 3
        finally:
            os.environ.pop("LOCAL_RANK", None)

    def cleanup(self) -> None:
        dist.init()
        dist.cleanup()
        assert dist.initialized() is False
        assert dist.state.initialized is False