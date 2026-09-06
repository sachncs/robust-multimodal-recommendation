"""Structured logging for morel.

`configure(dir, level, structured)` initializes the root logger with a JSON
formatter. `log(...)` writes one JSONL line per call for downstream parsing.

No module in morel uses `print()`. Every diagnostic goes through `get_logger`.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Format *record* as a single-line JSON string."""
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in payload or key.startswith("_"):
                continue
            if key in (
                "args",
                "msg",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "name",
                "taskName",
            ):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
        return json.dumps(payload, ensure_ascii=False)


@dataclass
class Config:
    """Logging configuration."""

    level: str = "INFO"
    structured: bool = True
    directory: Path | None = None


def configure(
    level: str = "INFO", directory: Path | str | None = None, structured: bool = True
) -> None:
    """Configure the root morel logger.

    Args:
        level: Log level name.
        directory: If set, also write a morel.log JSONL file.
        structured: Use JSON formatter; otherwise use a human-readable format.
    """
    root = logging.getLogger("morel")
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)
    formatter: logging.Formatter
    if structured:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)
    if directory is not None:
        path = Path(directory) / "morel.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    root.propagate = False


def get(name: str) -> logging.Logger:
    """Get a child logger of the morel root.

    Args:
        name: Dotted module path.

    Returns
    -------
        A logger configured at the morel root.
    """
    return logging.getLogger(f"morel.{name}")


def log(directory: Path | str, **metrics: Any) -> None:
    """Append a JSONL line of metrics to ``<directory>/metrics.jsonl``.

    Args:
        directory: Target run directory; created if missing.
        **metrics: Key-value pairs to record.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    line = {
        "time": datetime.now(tz=UTC).isoformat(),
        **metrics,
    }
    with (path / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, default=str, ensure_ascii=False))
        handle.write("\n")


__all__ = ["Config", "JsonFormatter", "configure", "get", "log"]
