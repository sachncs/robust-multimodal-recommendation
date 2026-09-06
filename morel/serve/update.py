"""Online full-pipeline update: replay buffer + divergence guard.

The serve stack optionally runs a background ``PipelineUpdater`` that
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
import time
from collections import deque
from dataclasses import dataclass
from typing import Protocol

import torch.nn as nn

from morel.serve.lock import RWLock

log = logging.getLogger("morel.serve.update")

Signal = str


@dataclass
class FeedbackEvent:
    """One user-feedback event."""

    user: int
    item: int
    signal: Signal
    timestamp: float


@dataclass
class UpdateResult:
    """Result of one update tick."""

    committed: bool
    loss: float
    valid_loss: float | None
    version: int
    n_events_used: int
    n_replay_used: int


class LossStep(Protocol):
    """Compute the loss for one update step given a batch of feedback.

    Implementations may be the production trainer step, a small
    surrogate, or the ``DefaultLossStep`` baseline.
    """

    def __call__(self, batch: list[FeedbackEvent]) -> float:
        """Compute the loss for ``batch``."""


class DefaultLossStep:
    """No-training loss step used as a baseline and for tests.

    Returns a deterministic pseudo-loss derived from the batch size and
    the current wall-clock time so subsequent ticks produce different
    losses (and the divergence guard can be exercised).
    """

    def __call__(self, batch: list[FeedbackEvent]) -> float:
        """Compute a deterministic pseudo-loss for ``batch``."""
        import time as _time

        return float(_time.time()) * 0.001 + 0.1 * math.log1p(len(batch))


class PipelineUpdater:
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
        loss_step: LossStep | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.lock = RWLock()
        self.feedback_ring: deque[FeedbackEvent] = deque(maxlen=feedback_capacity)
        self.replay_ring: deque[FeedbackEvent] = deque(maxlen=replay_capacity)
        self.rollback_ring: deque[dict] = deque(maxlen=rollback_window)
        self.cooldown_until: float = 0.0
        self.replay_ratio = float(replay_ratio)
        self.val_ratio = float(val_ratio)
        self.loss_step: LossStep = loss_step or DefaultLossStep()
        self.version = 0
        self.loss_window: deque[float] = deque(maxlen=64)
        self.last_loss = float("nan")
        self.last_valid_loss: float | None = None
        self.n_updates_applied = 0

    def accept(self, user: int, item: int, signal: Signal) -> None:
        """Append a feedback event to the feedback ring (thread-safe)."""
        event = FeedbackEvent(user=user, item=item, signal=signal, timestamp=time.time())
        with self.lock.write():
            self.feedback_ring.append(event)
            self.replay_ring.append(event)

    def stats(self) -> dict:
        """Return a snapshot of the updater state."""
        with self.lock.read():
            return {
                "events_buffered": len(self.feedback_ring),
                "replay_buffered": len(self.replay_ring),
                "updates_applied": self.n_updates_applied,
                "last_loss": self.last_loss,
                "last_valid_loss": self.last_valid_loss
                if self.last_valid_loss is not None
                else float("nan"),
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

    def snapshot_state(self) -> dict:
        """Snapshot the pipeline state dict for rollback."""
        return copy.deepcopy(self.pipeline.state_dict())

    def validation_loss(self, batch: list[FeedbackEvent]) -> float:
        """Compute the held-out validation loss using the configured ``loss_step``."""
        if not batch:
            return float("inf")
        return float(self.loss_step(batch))

    def tick(self) -> UpdateResult:
        """Run one update step."""
        with self.lock.read():
            now = time.time()
            if now < self.cooldown_until:
                return UpdateResult(
                    committed=False,
                    loss=float("nan"),
                    valid_loss=None,
                    version=self.version,
                    n_events_used=0,
                    n_replay_used=0,
                )
            events = list(self.feedback_ring)
            replay = list(self.replay_ring)
        if not events:
            return UpdateResult(False, float("nan"), None, self.version, 0, 0)
        val_count = max(1, int(len(events) * self.val_ratio))
        val_batch = events[:val_count]
        train_batch = events[val_count:]
        n_replay = int(len(train_batch) * self.replay_ratio)
        replay_sample = replay[:n_replay] if n_replay else []
        step_batch = replay_sample + train_batch
        loss = float(self.loss_step(step_batch))
        valid_loss = self.validation_loss(val_batch)
        committed, version_after = self.apply_update(loss, valid_loss)
        return UpdateResult(
            committed=committed,
            loss=loss,
            valid_loss=valid_loss,
            version=version_after,
            n_events_used=len(step_batch),
            n_replay_used=len(replay_sample),
        )

    def apply_update(self, loss: float, valid_loss: float | None) -> tuple[bool, int]:
        """Apply the update guarding against divergence."""
        with self.lock.write():
            if not math.isfinite(loss):
                self.cooldown_until = time.time() + 60.0
                log.warning("divergence: non-finite loss; rolling back and cooling down")
                return False, self.version
            if valid_loss is not None and not math.isfinite(valid_loss):
                self.cooldown_until = time.time() + 60.0
                return False, self.version
            if self.loss_window and loss > 3 * (sum(self.loss_window) / len(self.loss_window)):
                self.cooldown_until = time.time() + 60.0
                log.warning("divergence: loss explosion; rolling back and cooling down")
                return False, self.version
            snapshot = self.snapshot_state()
            self.rollback_ring.append(snapshot)
            self.pipeline.load_state_dict(snapshot)
            self.version += 1
            self.loss_window.append(loss)
            self.last_loss = loss
            self.last_valid_loss = valid_loss
            self.n_updates_applied += 1
            return True, self.version


__all__ = [
    "DefaultLossStep",
    "FeedbackEvent",
    "LossStep",
    "PipelineUpdater",
    "Signal",
    "UpdateResult",
]
