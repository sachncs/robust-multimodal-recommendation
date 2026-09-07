"""Tests for morel.app."""

from __future__ import annotations

from pathlib import Path

from morel.app import Benchmark, Experiment, Reproduce
from morel.core.config import Config


class Checker:
    """Aggregated test methods for this module."""

    def experiment(self, tmp_path: Path) -> None:
        config = Config()
        exp = Experiment(config=config, dir=tmp_path, items=10, users=4, epochs=1)
        result = exp.run()
        assert "duration" in result
        assert "dir" in result
        assert "cfg_hash" in result
        assert (tmp_path / "config.yaml").exists()
        assert (tmp_path / "manifest.json").exists()
        assert (tmp_path / "metrics.jsonl").exists()
        assert (tmp_path / "FIDELITY.md").exists()
        assert (tmp_path / "report.md").exists()

    def reproduce(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        Config().save(config_path)
        dir = tmp_path / "run"
        rep = Reproduce(path=config_path, dir=dir, items=10, users=4, epochs=1)
        result = rep.run()
        assert "duration" in result
        assert "cfg_hash" in result

    def benchmark(self, tmp_path: Path) -> None:
        config = Config()
        bench = Benchmark(config=config, dir=tmp_path, sizes=[8], epochs=1)
        result = bench.run()
        assert "results" in result
        assert "forward_s" in result["results"]
