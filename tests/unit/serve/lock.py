"""Concurrency contracts for the serve stack's reader-writer lock.

The lock existed and was covered only by single-threaded tests, which cannot
observe the property that matters: whether a writer makes progress while
readers keep arriving. It did not. Under sustained reader load a writer was
starved indefinitely, so a live model update would silently never be applied.

These tests run real threads. They are written to be decisive rather than
timing-sensitive: progress is asserted via events and barriers with generous
timeouts, and the mutual-exclusion checks record observed overlap rather than
sleeping and hoping.
"""

from __future__ import annotations

import threading
import time

import pytest

from morel.serve.lock import RWLock, reader, writer

# Generous enough that a correct lock always passes on a loaded CI machine,
# short enough that a starved writer fails rather than hanging the suite.
TIMEOUT = 10.0


class Checker:
    """Aggregated test methods for this module."""

    def once(self) -> None:
        """Readers must be concurrent, otherwise this is just a mutex."""
        lock = RWLock()
        readers = 4
        inside = threading.Barrier(readers, timeout=TIMEOUT)
        failures: list[BaseException] = []

        def read() -> None:
            try:
                with reader(lock):
                    inside.wait()
            except BaseException as exc:
                failures.append(exc)

        threads = [threading.Thread(target=read) for _ in range(readers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=TIMEOUT)

        assert not failures, f"readers could not be inside together: {failures}"
        assert lock.readers == 0

    def readers(self) -> None:
        """Regression: the writer never acquired the lock under sustained reads."""
        lock = RWLock()
        stop = threading.Event()

        def read_loop() -> None:
            while not stop.is_set():
                lock.read_lock()
                time.sleep(0.002)
                lock.read_unlock()

        threads = [threading.Thread(target=read_loop, daemon=True) for _ in range(8)]
        for thread in threads:
            thread.start()
        time.sleep(0.05)

        try:
            acquired = lock.write_lock(timeout=TIMEOUT)
            assert acquired, "writer was starved by the reader stream"
            lock.write_unlock()
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=TIMEOUT)

    def new(self) -> None:
        """Writer preference is the mechanism that prevents starvation."""
        lock = RWLock()
        lock.read_lock()

        writer_waiting = threading.Event()
        writer_done = threading.Event()

        def write() -> None:
            writer_waiting.set()
            lock.write_lock()
            lock.write_unlock()
            writer_done.set()

        thread = threading.Thread(target=write, daemon=True)
        thread.start()
        assert writer_waiting.wait(timeout=TIMEOUT)
        # Give the writer time to register itself as waiting.
        for _ in range(100):
            if lock.writers > 0:
                break
            time.sleep(0.01)
        assert lock.writers == 1

        assert not lock.read_lock(timeout=0.2), "a new reader jumped ahead of a waiting writer"

        lock.read_unlock()
        assert writer_done.wait(timeout=TIMEOUT)
        thread.join(timeout=TIMEOUT)

    def excludes(self) -> None:
        """No reader may be inside while a writer holds the lock."""
        lock = RWLock()
        overlaps: list[str] = []
        inside_write = threading.Event()
        write_unlock = threading.Event()

        def write() -> None:
            with writer(lock):
                inside_write.set()
                write_unlock.wait(timeout=TIMEOUT)

        def read() -> None:
            with reader(lock):
                if inside_write.is_set() and not write_unlock.is_set():
                    overlaps.append("reader overlapped an active writer")

        writer_thread = threading.Thread(target=write, daemon=True)
        writer_thread.start()
        assert inside_write.wait(timeout=TIMEOUT)

        reader_threads = [threading.Thread(target=read, daemon=True) for _ in range(4)]
        for thread in reader_threads:
            thread.start()
        time.sleep(0.1)
        write_unlock.set()
        for thread in reader_threads:
            thread.join(timeout=TIMEOUT)
        writer_thread.join(timeout=TIMEOUT)

        assert not overlaps, overlaps

    def serialised(self) -> None:
        """Two writers must never be inside at once."""
        lock = RWLock()
        concurrent = 0
        peak = 0
        guard = threading.Lock()
        errors: list[str] = []

        def write() -> None:
            nonlocal concurrent, peak
            for _ in range(20):
                with writer(lock):
                    with guard:
                        concurrent += 1
                        peak = max(peak, concurrent)
                        if concurrent > 1:
                            errors.append("two writers inside at once")
                    time.sleep(0.001)
                    with guard:
                        concurrent -= 1

        threads = [threading.Thread(target=write) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=TIMEOUT)

        assert not errors, errors
        assert peak == 1

    def load(self) -> None:
        """A leaked counter would wedge the lock for every later caller."""
        lock = RWLock()
        errors: list[BaseException] = []

        def read() -> None:
            try:
                for _ in range(50):
                    with reader(lock):
                        pass
            except BaseException as exc:
                errors.append(exc)

        def write() -> None:
            try:
                for _ in range(20):
                    with writer(lock):
                        pass
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=read) for _ in range(6)]
        threads += [threading.Thread(target=write) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=TIMEOUT)

        assert not errors, errors
        assert lock.readers == 0
        assert lock.writers == 0
        assert lock.writer is False

    def out(self) -> None:
        """A writer that gives up must not leave readers queued behind it."""
        lock = RWLock()
        lock.read_lock()

        assert lock.write_lock(timeout=0.05) is False
        assert lock.writers == 0, "a timed-out writer must deregister itself"

        lock.read_unlock()
        assert lock.read_lock(timeout=TIMEOUT) is True
        lock.read_unlock()

    def acquiring(self) -> None:
        lock = RWLock()
        lock.write_lock()
        try:
            assert lock.read_lock(timeout=0.05) is False
            assert lock.readers == 0
        finally:
            lock.write_unlock()

    def without(self) -> None:
        lock = RWLock()
        lock.read_lock()
        try:
            assert lock.write_lock(timeout=0.05) is False
            assert lock.writer is False
        finally:
            lock.read_unlock()

    def rejected(self) -> None:
        """Going negative would let a writer in while a reader is still inside."""
        with pytest.raises(RuntimeError, match="without holding a read lock"):
            RWLock().read_unlock()

    def check(self) -> None:
        with pytest.raises(RuntimeError, match="without holding the write lock"):
            RWLock().write_unlock()

    def exception(self) -> None:
        lock = RWLock()
        with pytest.raises(ValueError, match="boom"), reader(lock):
            raise ValueError("boom")
        assert lock.readers == 0

        with pytest.raises(ValueError, match="boom"), writer(lock):
            raise ValueError("boom")
        assert lock.writer is False
