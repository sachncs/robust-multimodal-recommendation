"""Tests for two-token auth, RWLock, and PipelineUpdater."""

from __future__ import annotations

import threading
import time
from typing import ClassVar

import pytest
import torch.nn as nn

from morel.serve.auth import (
    admin_enabled,
    assert_configured,
    read_enabled,
    require,
    token_for_scope,
)
from morel.serve.lock import RWLock
from morel.serve.update import PipelineUpdater


class TinyModel(nn.Module):
    """Small module used as a stand-in pipeline for updater tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)


# ---- Two-token auth ----








# ---- RWLock ----




# ---- PipelineUpdater ----


class Checker:
    """Aggregated test methods for this module."""

    def auth(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MOREL_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("MOREL_AUTH_TOKEN_ADMIN", raising=False)
        monkeypatch.setenv("MOREL_AUTH_TOKEN_READ", "read-tok")
        assert read_enabled() is True
        assert admin_enabled() is False
        assert token_for_scope("read") == "read-tok"
        assert token_for_scope("admin") is None

    def admin(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MOREL_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("MOREL_AUTH_TOKEN_READ", raising=False)
        monkeypatch.setenv("MOREL_AUTH_TOKEN_ADMIN", "admin-tok")
        assert read_enabled() is False
        assert admin_enabled() is True
        assert token_for_scope("read") is None
        assert token_for_scope("admin") == "admin-tok"

    def legacy(monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "MOREL_AUTH_TOKEN",
            "MOREL_AUTH_TOKEN_READ",
            "MOREL_AUTH_TOKEN_ADMIN",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("MOREL_AUTH_TOKEN", "legacy")
        assert read_enabled() is True
        assert admin_enabled() is True
        assert token_for_scope("read") == "legacy"
        assert token_for_scope("admin") == "legacy"

    def configured(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from morel.core.errors import ConfigError

        for var in (
            "MOREL_AUTH_TOKEN",
            "MOREL_AUTH_TOKEN_READ",
            "MOREL_AUTH_TOKEN_ADMIN",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("MOREL_AUTH_ENABLED", "1")
        with pytest.raises(ConfigError):
            assert_configured()

    def require() -> None:
        class Req:
            headers: ClassVar[dict[str, str]] = {}

        require(Req(), scope="read")

    def wrong(monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi import HTTPException

        monkeypatch.setenv("MOREL_AUTH_TOKEN_READ", "real-token")

        class Req:
            headers: ClassVar[dict[str, str]] = {"authorization": "Bearer wrong"}

        with pytest.raises(HTTPException):
            require(Req(), scope="read")

    def rwlock() -> None:
        lock = RWLock()
        errors: list[Exception] = []
        readers_in = 0
        readers_in_lock = threading.Lock()

        def reader() -> None:
            nonlocal readers_in
            try:
                with lock.read():
                    with readers_in_lock:
                        readers_in += 1
                    time.sleep(0.05)
                    with readers_in_lock:
                        readers_in -= 1
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def writer() -> None:
        lock = RWLock()
        counter = [0]
        counter_lock = threading.Lock()

        def writer() -> None:
            with lock.write():
                with counter_lock:
                    assert counter[0] == 0
                    counter[0] += 1
                time.sleep(0.02)
                with counter_lock:
                    counter[0] -= 1

        threads = [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert counter[0] == 0

    def updater() -> None:
        updater = PipelineUpdater(TinyModel())
        for i in range(8):
            updater.accept(user=i, item=i, signal="like")
        assert updater.stats()["events_buffered"] == 8
        assert updater.stats()["replay_buffered"] == 8

    def tick() -> None:
        updater = PipelineUpdater(TinyModel())
        result = updater.tick()
        assert result.committed is False

    def increments() -> None:
        updater = PipelineUpdater(
            TinyModel(),
            cooldown_seconds=0,
            loss_step=lambda batch: 0.5,
        )
        for i in range(32):
            updater.accept(user=i, item=i, signal="like")
        result = updater.tick()
        assert result.committed is True
        assert result.version == 1
        assert updater.stats()["updates_applied"] == 1

    def triggers() -> None:
        updater = PipelineUpdater(
            TinyModel(),
            cooldown_seconds=30,
            loss_step=lambda batch: float("nan"),
        )
        for i in range(16):
            updater.accept(user=i, item=i, signal="like")
        result = updater.tick()
        assert result.committed is False
        assert updater.cooldown_until > time.time()

    def rollback() -> None:
        updater = PipelineUpdater(
            TinyModel(),
            cooldown_seconds=0,
            loss_step=lambda batch: 0.5,
        )
        for i in range(32):
            updater.accept(user=i, item=i, signal="like")
        updater.tick()
        updater.tick()
        assert updater.version == 2
        version = updater.rollback(steps=1)
        assert version == 1
        assert updater.version == 1

    def default() -> None:
        """LossStep Protocol: DefaultLossStep and a custom callable both work."""
        from morel.serve.update import DefaultLossStep

        base = PipelineUpdater(TinyModel(), cooldown_seconds=0)
        assert isinstance(base.loss_step, DefaultLossStep)
        custom = PipelineUpdater(
            TinyModel(),
            cooldown_seconds=0,
            loss_step=lambda batch: 0.42,
        )
        for i in range(8):
            custom.accept(user=i, item=i, signal="like")
        result = custom.tick()
        assert result.committed is True
        assert abs(result.loss - 0.42) < 1e-6
