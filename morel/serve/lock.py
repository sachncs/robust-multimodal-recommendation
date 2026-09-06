"""Reader-writer lock for the live serve pipeline.

The serve stack runs inference requests concurrently while the
``PipelineUpdater`` runs periodic update steps. A reader-writer lock
ensures readers never see a half-applied update, and writers are
serialised.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator


class RWLock:
    """A simple reader-writer lock.

    Multiple readers can hold the lock concurrently. Writers are
    exclusive.
    """

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.active_readers = 0
        self.active_writer = False

    def acquire_read(self, timeout: float | None = None) -> bool:
        """Block until a read lock is acquired, or return False on timeout."""
        with self.condition:
            if timeout is None:
                while self.active_writer:
                    self.condition.wait()
                self.active_readers += 1
                return True
            self.condition.wait_for(
                lambda: not self.active_writer, timeout=timeout
            )
            if self.active_writer:
                return False
            self.active_readers += 1
            return True

    def release_read(self) -> None:
        with self.condition:
            self.active_readers -= 1
            if self.active_readers == 0:
                self.condition.notify_all()

    def acquire_write(self, timeout: float | None = None) -> bool:
        with self.condition:
            if timeout is None:
                while self.active_writer or self.active_readers > 0:
                    self.condition.wait()
                self.active_writer = True
                return True
            self.condition.wait_for(
                lambda: not self.active_writer and self.active_readers == 0,
                timeout=timeout,
            )
            if self.active_writer or self.active_readers > 0:
                return False
            self.active_writer = True
            return True

    def release_write(self) -> None:
        with self.condition:
            self.active_writer = False
            self.condition.notify_all()

    def read(self) -> "ReadGuard":
        """Return a context manager that acquires/releases a read lock."""
        return ReadGuard(self)

    def write(self) -> "WriteGuard":
        """Return a context manager that acquires/releases a write lock."""
        return WriteGuard(self)


class ReadGuard:
    """Context manager returned by :meth:`RWLock.read`."""

    def __init__(self, lock: RWLock) -> None:
        self.lock = lock

    def __enter__(self) -> None:
        self.lock.acquire_read()

    def __exit__(self, *exc: object) -> None:
        self.lock.release_read()


class WriteGuard:
    """Context manager returned by :meth:`RWLock.write`."""

    def __init__(self, lock: RWLock) -> None:
        self.lock = lock

    def __enter__(self) -> None:
        self.lock.acquire_write()

    def __exit__(self, *exc: object) -> None:
        self.lock.release_write()


@contextmanager
def reader(lock: RWLock) -> Iterator[None]:
    """Context manager for a read lock."""
    lock.acquire_read()
    try:
        yield None
    finally:
        lock.release_read()


@contextmanager
def writer(lock: RWLock) -> Iterator[None]:
    """Context manager for a write lock."""
    lock.acquire_write()
    try:
        yield None
    finally:
        lock.release_write()


__all__ = ["RWLock", "ReadGuard", "WriteGuard", "reader", "writer"]
