"""JSONL metrics monitor."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Monitor:
    """Append ``(time, **metrics)`` lines to ``metrics.jsonl``."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "metrics.jsonl"

    def log(self, step: int | None = None, **metrics: Any) -> None:
        """Append one JSONL line."""
        record: dict[str, Any] = {"time": time.time()}
        if step is not None:
            record["step"] = int(step)
        record.update(metrics)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str, ensure_ascii=False))
            handle.write("\n")

    def latest(self) -> dict[str, Any] | None:
        """Return the most recent record or None if empty."""
        if not self.path.exists():
            return None
        with self.path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
        if not lines:
            return None
        record: dict[str, Any] = json.loads(lines[-1])
        return record


__all__ = ["Monitor"]
