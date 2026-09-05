"""Tests for morel.app."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from morel.app import Benchmark, Experiment, Reproduce
from morel.core.config import Config


def test_experiment_run(tmp_path: Path) -> None:
    config = Config()
    exp = Experiment(config=config, run_dir=tmp_path, items=10, users=4, epochs=1)
    result = exp.run()
    assert "duration" in result
    assert "run_dir" in result
    assert "config_hash" in result
    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "FIDELITY.md").exists()
    assert (tmp_path / "report.md").exists()


def test_reproduce_run(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    Config().to_yaml(config_path)
    run_dir = tmp_path / "run"
    rep = Reproduce(config_path=config_path, run_dir=run_dir, items=10, users=4, epochs=1)
    result = rep.run()
    assert "duration" in result
    assert "config_hash" in result


def test_benchmark_run(tmp_path: Path) -> None:
    config = Config()
    bench = Benchmark(config=config, run_dir=tmp_path, sizes=[8], epochs=1)
    result = bench.run()
    assert "results" in result
    assert "forward_s" in result["results"]
