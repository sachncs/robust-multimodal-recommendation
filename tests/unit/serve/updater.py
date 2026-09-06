"""Concurrency contracts for the live updater.

Two locking defects had to be fixed together. Without writer preference the
model updater was starved by inference traffic and updates never applied. With
writer preference but a single lock, appending a feedback event took the model
write lock, and inference collapsed instead: model reads fell to single digits
per second under eight feedback threads.

The updater therefore keeps model state and the feedback buffers under separate
locks. These tests assert both directions of progress.
"""

from __future__ import annotations

import threading
import time

import torch.nn as nn

from morel.serve.update import Updater

TIMEOUT = 10.0

#: Before the locks were separated this benchmark produced single-digit reads
#: per second; afterwards it produces hundreds of thousands. Any floor in
#: between is decisive, so this is set low enough to be safe on slow hardware.
MIN_READS = 200


def updater() -> Updater:
    """Return an updater over a trivial model."""
    return Updater(nn.Linear(4, 4))


class Checker:
    """Aggregated test methods for this module."""

    def load(self) -> None:
        """Regression: accept() took the model write lock and starved inference."""
        live = updater()
        stop = threading.Event()
        reads = 0

        def feedback_loop() -> None:
            while not stop.is_set():
                live.accept(1, 2, "click")

        threads = [threading.Thread(target=feedback_loop, daemon=True) for _ in range(8)]
        for thread in threads:
            thread.start()
        try:
            deadline = time.time() + 0.5
            while time.time() < deadline:
                with live.lock.read():
                    pass
                reads += 1
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=TIMEOUT)

        assert reads > MIN_READS, f"only {reads} model reads completed under feedback load"

    def concurrency(self) -> None:
        """Every accepted event must land in both rings."""
        live = updater()
        per_thread = 200
        threads = [
            threading.Thread(target=lambda: [live.accept(1, 2, "click") for _ in range(per_thread)])
            for _ in range(6)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=TIMEOUT)

        assert len(live.feedback_ring) == per_thread * 6
        assert len(live.replay_ring) == per_thread * 6

    def arrives(self) -> None:
        """The other direction: the updater must not be starved either."""
        live = updater()
        stop = threading.Event()

        def feedback_loop() -> None:
            while not stop.is_set():
                live.accept(1, 2, "click")

        threads = [threading.Thread(target=feedback_loop, daemon=True) for _ in range(8)]
        for thread in threads:
            thread.start()
        time.sleep(0.05)
        try:
            assert live.lock.write_acquire(timeout=TIMEOUT), "model update was starved"
            live.lock.write_release()
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=TIMEOUT)

    def feedback(self) -> None:
        """stats() must not raise or report nonsense while events arrive."""
        live = updater()
        stop = threading.Event()
        failures: list[BaseException] = []

        def feedback_loop() -> None:
            while not stop.is_set():
                live.accept(1, 2, "click")

        threads = [threading.Thread(target=feedback_loop, daemon=True) for _ in range(4)]
        for thread in threads:
            thread.start()
        try:
            for _ in range(500):
                try:
                    snapshot = live.stats()
                    assert snapshot["events_buffered"] >= 0
                    assert snapshot["replay_buffered"] >= 0
                    assert snapshot["current_version"] >= 0
                except BaseException as exc:
                    failures.append(exc)
                    break
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=TIMEOUT)

        assert not failures, failures

    def use(self) -> None:
        """A leaked counter would wedge every later caller."""
        live = updater()
        threads = [
            threading.Thread(target=lambda: [live.accept(1, 2, "click") for _ in range(50)])
            for _ in range(4)
        ]
        threads += [
            threading.Thread(target=lambda: [live.stats() for _ in range(50)]) for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=TIMEOUT)

        assert live.lock.active_readers == 0
        assert live.lock.waiting_writers == 0
        assert live.lock.active_writer is False
        assert not live.buffer_lock.locked()