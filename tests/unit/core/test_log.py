"""Tests for morel.core.log."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from morel.core.log import configure, get, log


def test_configure_adds_root_handler(caplog: object) -> None:
    configure(level="DEBUG", structured=True, directory=None)
    logger = get("test")
    logger.info("hello")
    root = logging.getLogger("morel")
    assert root.handlers


def test_json_formatter_emits_json() -> None:
    configure(level="INFO", structured=True, directory=None)
    from io import StringIO

    buf = StringIO()
    handler = logging.StreamHandler(buf)
    from morel.core.log import JsonFormatter

    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("morel.test_json")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.info("hello", extra={"metric": 1.5})
    payload = json.loads(buf.getvalue().strip())
    assert payload["message"] == "hello"
    assert payload["metric"] == 1.5


def test_log_appends_metrics(tmp_path: Path) -> None:
    configure(level="INFO", structured=False, directory=tmp_path)
    log(tmp_path, loss=0.1, step=1)
    log(tmp_path, loss=0.05, step=2)
    path = tmp_path / "metrics.jsonl"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["loss"] == 0.1
    assert json.loads(lines[1])["step"] == 2
