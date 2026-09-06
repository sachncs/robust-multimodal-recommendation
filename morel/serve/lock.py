"""Reader-writer lock for the live serve pipeline.

The serve stack runs inference requests concurrently while the
``PipelineUpdater`` runs periodic update steps. A reader-writer lock
ensures readers never see a half-applied update, and writers are
serialised.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class RWLock:
    """A reader-writer lock with writer preference.

    Multiple readers hold the lock concurrently; writers are exclusive.

    Once a writer is waiting, new readers queue behind it. Without that, a
    stream of overlapping readers keeps ``active_readers`` above zero forever
    and a writer never runs: under sustained inference traffic the model
    updater would be starved indefinitely and updates would silently never be
    applied. Readers therefore wait at most for the duration of one update,
    and writers are guaranteed to make progress.

    The lock is not reentrant. A thread already holding a read lock must not
    ask for the write lock, and vice versa; doing so deadlocks.
    """

    def __init__(self) -> None:
        """Create an unheld lock."""
        self.condition = threading.Condition()
        self.active_readers = 0
        self.active_writer = False
        self.waiting_writers = 0

    def acquire_read(self, timeout: float | None = None) -> bool:
        """Acquire a read lock, waiting for any active or pending writer.

        Args:
            timeout: Seconds to wait, or ``None`` to wait indefinitely.

        Returns
        -------
            ``True`` if the lock was acquired, ``False`` on timeout.
        """
        with self.condition:
            if timeout is None:
                while self.active_writer or self.waiting_writers > 0:
                    self.condition.wait()
            elif not self.condition.wait_for(
                lambda: not self.active_writer and self.waiting_writers == 0,
                timeout=timeout,
            ):
                return False
            self.active_readers += 1
            return True

    def release_read(self) -> None:
        """Release a previously acquired read lock.

        Raises
        ------
            RuntimeError: If no read lock is held. Silently going negative
                would let a later writer proceed while a reader is still
                inside the critical section.
        """
        with self.condition:
            if self.active_readers <= 0:
                raise RuntimeError("release_read called without holding a read lock")
            self.active_readers -= 1
            if self.active_readers == 0:
                self.condition.notify_all()

    def acquire_write(self, timeout: float | None = None) -> bool:
        """Acquire the exclusive write lock.

        Registers as a waiting writer first, which holds off new readers so
        that this call cannot be starved by a continuous read stream.

        Args:
            timeout: Seconds to wait, or ``None`` to wait indefinitely.

        Returns
        -------
            ``True`` if the lock was acquired, ``False`` on timeout.
        """
        with self.condition:
            self.waiting_writers += 1
            try:
                if timeout is None:
                    while self.active_writer or self.active_readers > 0:
                        self.condition.wait()
                elif not self.condition.wait_for(
                    lambda: not self.active_writer and self.active_readers == 0,
                    timeout=timeout,
                ):
                    return False
                self.active_writer = True
                return True
            finally:
                self.waiting_writers -= 1
                if not self.active_writer:
                    # Gave up: wake the readers that were queued behind us,
                    # otherwise they wait for a writer that is no longer coming.
                    self.condition.notify_all()

    def release_write(self) -> None:
        """Release a previously acquired write lock.

        Raises
        ------
            RuntimeError: If the write lock is not held.
        """
        with self.condition:
            if not self.active_writer:
                raise RuntimeError("release_write called without holding the write lock")
            self.active_writer = False
            self.condition.notify_all()

    def read(self) -> ReadGuard:
        """Return a context manager that acquires/releases a read lock."""
        return ReadGuard(self)

    def write(self) -> WriteGuard:
        """Return a context manager that acquires/releases a write lock."""
        return WriteGuard(self)


class ReadGuard:
    """Context manager returned by :meth:`RWLock.read`."""

    def __init__(self, lock: RWLock) -> None:
        self.lock = lock

    def __enter__(self) -> None:
        """Acquire the read lock."""
        self.lock.acquire_read()

    def __exit__(self, *exc: object) -> None:
        """Release the read lock."""
        self.lock.release_read()


class WriteGuard:
    """Context manager returned by :meth:`RWLock.write`."""

    def __init__(self, lock: RWLock) -> None:
        self.lock = lock

    def __enter__(self) -> None:
        """Acquire the write lock."""
        self.lock.acquire_write()

    def __exit__(self, *exc: object) -> None:
        """Release the write lock."""
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
