"""The CLI must read the configuration rather than hardcoding its behaviour.

``configure_log`` was called with a fixed level, the eval commands used
hardcoded cutoffs and mask ratios, and ``morel serve`` had its own host and
port defaults. Every one of those config fields was inert.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from morel.cli import configure_logging, run_eval
from morel.core.config import Config


def write(tmp_path: Path, **payload: Any) -> Path:
    """Write a config YAML and return its path."""
    path = tmp_path / "config.yaml"
    Config.from_dict(payload).to_yaml(path)
    return path


class Checker:
    """Aggregated test methods for this module."""

    def rank(tmp_path: Path, capsys: Any) -> None:
        path = write(tmp_path, eval={"ks": [3, 7]})
        assert run_eval(["rank", "--config", str(path)]) == 0

        out = capsys.readouterr().out
        assert "recall@3=" in out
        assert "recall@7=" in out
        assert "recall@10=" not in out, "the hardcoded k=10 must be gone"

    def robustness(tmp_path: Path, capsys: Any) -> None:
        path = write(tmp_path, eval={"robustness": [0.2, 0.8]})
        assert run_eval(["robustness", "--config", str(path)]) == 0
        out = capsys.readouterr().out
        assert "0.2" in out
        assert "0.8" in out
        assert "0.7" not in out, "the hardcoded ratio list must be gone"

    def defaults(capsys: Any) -> None:
        assert run_eval(["rank"]) == 0
        out = capsys.readouterr().out
        for k in Config().eval.ks:
            assert f"recall@{k}=" in out

    def logging(tmp_path: Path) -> None:
        import logging

        path = write(tmp_path, log={"level": "ERROR", "structured": True})
        configure_logging(["--config", str(path)])
        assert logging.getLogger("morel").level == logging.ERROR

        configure_logging([])
        assert logging.getLogger("morel").level == logging.getLevelName(Config().log.level)

    def survives(tmp_path: Path) -> None:
        """The logger must come up so the user can be told the config is broken."""
        import logging

        bad = tmp_path / "bad.yaml"
        bad.write_text("{not: valid: yaml", encoding="utf-8")
        configure_logging(["--config", str(bad)])
        assert logging.getLogger("morel").level == logging.INFO

    def serve() -> None:
        result = subprocess.run(
            [sys.executable, "-m", "morel", "serve", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        for flag in ("--host", "--port", "--workers"):
            assert flag in result.stdout, f"{flag} missing from serve --help"
        assert "serve.host" in result.stdout

    def uses(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        class FakeUvicorn:
            @staticmethod
            def run(app: Any, **kwargs: Any) -> None:
                captured.update(kwargs)

        monkeypatch.setitem(sys.modules, "uvicorn", FakeUvicorn)
        path = write(tmp_path, serve={"host": "127.0.0.1", "port": 9111, "workers": 3})

        from morel.cli import serve_inference

        assert serve_inference(["--config", str(path)]) == 0
        assert captured["host"] == "127.0.0.1"
        assert captured["port"] == 9111
        assert captured["workers"] == 3

    def flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class FakeUvicorn:
            @staticmethod
            def run(app: Any, **kwargs: Any) -> None:
                captured.update(kwargs)

        monkeypatch.setitem(sys.modules, "uvicorn", FakeUvicorn)
        path = write(tmp_path, serve={"host": "127.0.0.1", "port": 9111})

        from morel.cli import serve_inference

        assert serve_inference(["--config", str(path), "--port", "1234"]) == 0
        assert captured["host"] == "127.0.0.1", "unspecified flags still come from config"
        assert captured["port"] == 1234, "an explicit flag must win"
