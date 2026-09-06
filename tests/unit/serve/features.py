"""Tests for two-token auth, RWLock, and Updater."""

from __future__ import annotations

import threading
import time
from typing import ClassVar

import pytest
import torch.nn as nn

from morel.serve.auth import (
    assert_set,
    is_admin,
    is_read,
    require,
    token,
)
from morel.serve.lock import RWLock
from morel.serve.update import Updater


class Tiny(nn.Module):
    """Small module used as a stand-in pipeline for updater tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)


# ---- Two-token auth ----


# ---- RWLock ----


# ---- Updater ----


class Checker:
    """Aggregated test methods for this module."""

    def token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MOREL_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("MOREL_AUTH_TOKEN_ADMIN", raising=False)
        monkeypatch.setenv("MOREL_AUTH_TOKEN_READ", "read-tok")
        assert is_read() is True
        assert is_admin() is False
        assert token("read") == "read-tok"
        assert token("admin") is None

    def only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MOREL_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("MOREL_AUTH_TOKEN_READ", raising=False)
        monkeypatch.setenv("MOREL_AUTH_TOKEN_ADMIN", "admin-tok")
        assert is_read() is False
        assert is_admin() is True
        assert token("read") is None
        assert token("admin") == "admin-tok"

    def both(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "MOREL_AUTH_TOKEN",
            "MOREL_AUTH_TOKEN_READ",
            "MOREL_AUTH_TOKEN_ADMIN",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("MOREL_AUTH_TOKEN", "legacy")
        assert is_read() is True
        assert is_admin() is True
        assert token("read") == "legacy"
        assert token("admin") == "legacy"

    def without(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from morel.core.errors import Cfg

        for var in (
            "MOREL_AUTH_TOKEN",
            "MOREL_AUTH_TOKEN_READ",
            "MOREL_AUTH_TOKEN_ADMIN",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("MOREL_AUTH_ENABLED", "1")
        with pytest.raises(Cfg):
            assert_set()

    def noop(self) -> None:
        class Req:
            headers: ClassVar[dict[str, str]] = {}

        require(Req(), scope="read")

    def raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi import HTTPException

        monkeypatch.setenv("MOREL_AUTH_TOKEN_READ", "real-token")

        class Req:
            headers: ClassVar[dict[str, str]] = {"authorization": "Bearer wrong"}

        with pytest.raises(HTTPException):
            require(Req(), scope="read")

    def other(self) -> None:
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

    def writers(self) -> None:
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

    def buffer(self) -> None:
        updater = Updater(Tiny())
        for i in range(8):
            updater.accept(user=i, item=i, signal="like")
        assert updater.stats()["events_buffered"] == 8
        assert updater.stats()["replay_buffered"] == 8

    def returns(self) -> None:
        updater = Updater(Tiny())
        result = updater.tick()
        assert result.committed is False

    def commit(self) -> None:
        updater = Updater(
            Tiny(),
            cooldown_seconds=0,
            loss_step=lambda batch: 0.5,
        )
        for i in range(32):
            updater.accept(user=i, item=i, signal="like")
        result = updater.tick()
        assert result.committed is True
        assert result.version == 1
        assert updater.stats()["updates_applied"] == 1

    def loss(self) -> None:
        updater = Updater(
            Tiny(),
            cooldown_seconds=30,
            loss_step=lambda batch: float("nan"),
        )
        for i in range(16):
            updater.accept(user=i, item=i, signal="like")
        result = updater.tick()
        assert result.committed is False
        assert updater.cooldown_until > time.time()

    def version(self) -> None:
        updater = Updater(
            Tiny(),
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

    def polymorphic(self) -> None:
        """Step Protocol: DefaultStp and a custom callable both work."""
        from morel.serve.update import DefaultStp

        base = Updater(Tiny(), cooldown_seconds=0)
        assert isinstance(base.loss_step, DefaultStp)
        custom = Updater(
            Tiny(),
            cooldown_seconds=0,
            loss_step=lambda batch: 0.42,
        )
        for i in range(8):
            custom.accept(user=i, item=i, signal="like")
        result = custom.tick()
        assert result.committed is True
        assert abs(result.loss - 0.42) < 1e-6
