"""Online full-pipeline update: replay buffer + divergence guard.

The serve stack optionally runs a background ``Updater`` that
applies periodic updates to both the completion and the recommendation
stages using:

- a bounded ``feedback_ring`` of recent feedback events
- a bounded ``replay_ring`` of replayed events
- a divergence guard that rolls back when validation loss explodes
- a rollback ring of bounded depth for manual rollback
"""

from __future__ import annotations

import copy
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

import torch.nn as nn

from morel.serve.lock import RWLock

log = logging.getLogger("morel.serve.update")

Signal = str


@dataclass
class Event:
    """One user-feedback event."""

    user: int
    item: int
    signal: Signal
    timestamp: float


@dataclass
class Outcome:
    """Result of one update tick."""

    committed: bool
    loss: float
    valid_loss: float | None
    version: int
    n_events_used: int
    n_replay_used: int


class Step(Protocol):
    """Compute the loss for one update step given a batch of feedback.

    Implementations may be the production trainer step, a small
    surrogate, or the ``Default`` baseline.
    """

    def __call__(self, batch: list[Event]) -> float:
        """Compute the loss for ``batch``."""


class Default:
    """No-training loss step used as a baseline and for tests.

    Returns a deterministic pseudo-loss derived from the batch size and
    the current wall-clock time so subsequent ticks produce different
    losses (and the divergence guard can be exercised).
    """

    def __call__(self, batch: list[Event]) -> float:
        """Compute a deterministic pseudo-loss for ``batch``."""
        import time as _time

        return float(_time.time()) * 0.001 + 0.1 * math.log1p(len(batch))


class Updater:
    """Background updater that calls :meth:`tick` periodically.

    Args
    ----
    pipeline : nn.Module
        The live Pipeline. Snapshotting uses ``copy.deepcopy(state_dict())``.
    feedback_capacity : int
        Max events in the feedback ring.
    replay_capacity : int
        Max events in the replay ring.
    rollback_window : int
        Max number of state snapshots kept for rollback.
    cooldown_seconds : float
        Seconds to skip updates after a divergence event.
    replay_ratio : float
        Fraction of a batch drawn from the replay ring vs. feedback.
    val_ratio : float
        Fraction of the feedback ring reserved for held-out validation.
    """

    def __init__(
        self,
        pipeline: nn.Module,
        *,
        feedback_capacity: int = 10_000,
        replay_capacity: int = 10_000,
        rollback_window: int = 3,
        cooldown_seconds: float = 60.0,
        replay_ratio: float = 0.3,
        val_ratio: float = 0.1,
        loss_step: Step | None = None,
    ) -> None:
        self.pipeline = pipeline
        # Two locks for two concerns. ``lock`` guards model state -- weights,
        # version, rollback history -- where readers must never observe a
        # half-applied update. ``buffer_lock`` guards only the feedback and
        # replay rings.
        #
        # Appending a feedback event used to take the model write lock, which
        # made every event exclusive against every inference request. With a
        # correctly writer-preferring RWLock that collapsed read throughput to
        # about 4% of its unloaded rate. Buffer appends are not model updates
        # and do not need the model lock.
        self.lock = RWLock()
        self.sync = threading.Lock()
        self.ring: deque[Event] = deque(maxlen=feedback_capacity)
        self.replay_ring: deque[Event] = deque(maxlen=replay_capacity)
        self.rollback_ring: deque[dict[str, Any]] = deque(maxlen=rollback_window)
        self.cooldown_until: float = 0.0
        self.replay_ratio = float(replay_ratio)
        self.val = float(val_ratio)
        self.loss_step: Step = loss_step or Default()
        self.version = 0
        self.window: deque[float] = deque(maxlen=64)
        self.last = float("nan")
        self.valid: float | None = None
        self.updates = 0

    def accept(self, user: int, item: int, signal: Signal) -> None:
        """Append a feedback event to the feedback ring (thread-safe)."""
        event = Event(user=user, item=item, signal=signal, timestamp=time.time())
        with self.sync:
            self.ring.append(event)
            self.replay_ring.append(event)

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of the updater state."""
        # Take the buffer lock first and release it before the model lock, so
        # the two are always acquired in the same order and cannot deadlock.
        with self.sync:
            events_buffered = len(self.ring)
            replay_buffered = len(self.replay_ring)
        with self.lock.read():
            return {
                "events_buffered": events_buffered,
                "replay_buffered": replay_buffered,
                "updates_applied": self.updates,
                "last_loss": self.last,
                "valid_loss": self.valid if self.valid is not None else float("nan"),
                "current_version": self.version,
                "cooldown_until": self.cooldown_until,
            }

    def rollback(self, steps: int = 1) -> int:
        """Roll back ``steps`` versions. Returns the new version (-1 if none)."""
        with self.lock.write():
            for _ in range(steps):
                if not self.rollback_ring:
                    self.version = -1
                    return self.version
                self.rollback_ring.pop()
            if self.rollback_ring:
                self.version -= 1
                snapshot = self.rollback_ring[-1]
            else:
                self.version = -1
                return self.version
        self.pipeline.load_state_dict(snapshot)
        return self.version

    def snapshot(self) -> dict[str, Any]:
        """Snapshot the pipeline state dict for rollback."""
        return copy.deepcopy(self.pipeline.state_dict())

    def assess(self, batch: list[Event]) -> float:
        """Compute the held-out validation loss using the configured ``loss_step``."""
        if not batch:
            return float("inf")
        return float(self.loss_step(batch))

    def tick(self) -> Outcome:
        """Run one update step."""
        with self.lock.read():
            now = time.time()
            if now < self.cooldown_until:
                return Outcome(
                    committed=False,
                    loss=float("nan"),
                    valid_loss=None,
                    version=self.version,
                    n_events_used=0,
                    n_replay_used=0,
                )
        with self.sync:
            events = list(self.ring)
            replay = list(self.replay_ring)
        if not events:
            return Outcome(False, float("nan"), None, self.version, 0, 0)
        val_count = max(1, int(len(events) * self.val))
        val_batch = events[:val_count]
        train_batch = events[val_count:]
        n_replay = int(len(train_batch) * self.replay_ratio)
        replay_sample = replay[:n_replay] if n_replay else []
        step_batch = replay_sample + train_batch
        loss = float(self.loss_step(step_batch))
        valid_loss = self.assess(val_batch)
        committed, version_after = self.apply(loss, valid_loss)
        return Outcome(
            committed=committed,
            loss=loss,
            valid_loss=valid_loss,
            version=version_after,
            n_events_used=len(step_batch),
            n_replay_used=len(replay_sample),
        )

    def apply(self, loss: float, valid_loss: float | None) -> tuple[bool, int]:
        """Apply the update guarding against divergence."""
        with self.lock.write():
            if not math.isfinite(loss):
                self.cooldown_until = time.time() + 60.0
                log.warning("divergence: non-finite loss; rolling back and cooling down")
                return False, self.version
            if valid_loss is not None and not math.isfinite(valid_loss):
                self.cooldown_until = time.time() + 60.0
                return False, self.version
            if self.window and loss > 3 * (sum(self.window) / len(self.window)):
                self.cooldown_until = time.time() + 60.0
                log.warning("divergence: loss explosion; rolling back and cooling down")
                return False, self.version
            snapshot = self.snapshot()
            self.rollback_ring.append(snapshot)
            self.pipeline.load_state_dict(snapshot)
            self.version += 1
            self.window.append(loss)
            self.last = loss
            self.valid = valid_loss
            self.updates += 1
            return True, self.version


__all__ = [
    "Default",
    "Event",
    "Outcome",
    "Signal",
    "Step",
    "Updater",
]
