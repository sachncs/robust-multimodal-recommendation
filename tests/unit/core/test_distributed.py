"""Tests for morel.core.distributed."""

from __future__ import annotations

import pytest

from morel.core import distributed as dist


@pytest.fixture(autouse=True)
def reset_dist_state():
    """Reset module-level distributed state before each test."""
    dist.state.initialized = False
    dist.state.backend = None
    yield
    dist.state.initialized = False
    dist.state.backend = None


def test_default_world_size_is_one() -> None:
    assert dist.world_size() == 1


def test_default_rank_is_zero() -> None:
    assert dist.rank() == 0


def test_default_is_rank_zero() -> None:
    assert dist.is_rank_zero() is True


def test_init_single_process_returns_rank_zero() -> None:
    info = dist.init()
    assert info["rank"] == 0
    assert info["world_size"] == 1
    assert dist.is_initialized() is True


def test_init_is_idempotent() -> None:
    dist.init()
    dist.init()
    assert dist.is_initialized() is True
    assert dist.rank() == 0


def test_barrier_is_noop_in_single_process() -> None:
    dist.barrier()
    assert dist.is_rank_zero() is True


def test_reduce_mean_passthrough_in_single_process() -> None:
    dist.init()
    assert dist.reduce_mean(7.5) == 7.5


def test_local_rank_reads_env() -> None:
    import os

    os.environ["LOCAL_RANK"] = "3"
    try:
        assert dist.local_rank() == 3
    finally:
        os.environ.pop("LOCAL_RANK", None)


def test_cleanup_resets_state() -> None:
    dist.init()
    dist.cleanup()
    assert dist.is_initialized() is False
    assert dist.state.initialized is False
