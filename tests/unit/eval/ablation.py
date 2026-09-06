"""Ablation conditions must be expressible, runnable, and actually measure something.

``config.eval.ablations`` named three conditions that nothing could act on. The
names were inert strings, and the completion stage's output never reached the
ranker, so even once the conditions were runnable every one of them scored
identically.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from morel.app import Ablate
from morel.core.config import Config
from morel.core.errors import Cfg
from morel.eval import ABLATIONS, BASELINE, ablate, conditions


def small(**overrides: Any) -> Config:
    """Return a config small enough to run a sweep quickly."""
    payload: dict[str, Any] = {
        "encode": {"hidden": 16, "pe": 4, "layers": 1, "heads": 2},
        "codebook": {"size": 16},
        "recommend": {"kind": "light", "embed": 8, "layers": 2},
        "retrieve": {"anchors": 3, "iters": 2},
        "eval": {"ks": [5]},
    }
    payload.update(overrides)
    return Config.parse(payload)


class Checker:
    """Aggregated test methods for this module."""

    def registered(self) -> None:
        assert set(ABLATIONS) == {"noretry", "nope", "nobook"}

    def all(self) -> None:
        """A configured condition that is not registered would fail mid-sweep."""
        for name in Config().eval.ablations:
            assert name in ABLATIONS, f"{name} is named in eval.ablations but not registered"

    def baseline(self) -> None:
        config = small(eval={"ks": [5], "ablations": ["nobook"]})
        assert conditions(config) == (BASELINE, "nobook")

    def unchanged(self) -> None:
        config = small()
        assert ablate(config, BASELINE) is config

    def context(self) -> None:
        assert ablate(small(), "noretry").retrieve.kind == "none"

    def encoding(self) -> None:
        assert ablate(small(), "nope").encode.pe == 0

    def quantization(self) -> None:
        assert ablate(small(), "nobook").codebook.kind == "identity"

    def the(self) -> None:
        config = small()
        before = replace(config)
        for name in ABLATIONS:
            ablate(config, name)
        assert config == before

    def component(self) -> None:
        config = small()
        ablated = ablate(config, "nope")
        assert ablated.codebook == config.codebook
        assert ablated.retrieve == config.retrieve
        assert ablated.recommend == config.recommend
        assert ablated.encode.pe != config.encode.pe

    def rejected(self) -> None:
        with pytest.raises(Cfg, match="unknown ablation 'unknown'; available: "):
            ablate(small(), "unknown")

    def cutoff(self, tmp_path: Path) -> None:
        config = small(eval={"ks": [5, 10], "ablations": ["nope", "nobook"]})
        result = Ablate(config=config, run_dir=tmp_path, items=30, users=10).run()

        assert set(result["metrics"]) == {"recall@5", "ndcg@5", "recall@10", "ndcg@10"}
        for values in result["metrics"].values():
            assert set(values) == {BASELINE, "nope", "nobook"}
            for value in values.values():
                assert 0.0 <= value <= 1.0

    def metrics(self, tmp_path: Path) -> None:
        """Regression: completion output never reached the ranker, so all conditions tied.

        A sweep where every condition scores identically measures nothing, and
        would have been reported as though it did.
        """
        config = small(eval={"ks": [5], "ablations": ["noretry", "nobook"]})
        result = Ablate(config=config, run_dir=tmp_path, items=30, users=10).run()

        recall = result["metrics"]["recall@5"]
        assert len({round(v, 6) for v in recall.values()}) > 1, (
            f"every condition scored the same: {recall}"
        )

    def reproducible(self, tmp_path: Path) -> None:
        config = small()
        first = Ablate(config=config, run_dir=tmp_path / "a", items=30, users=10).run()
        second = Ablate(config=config, run_dir=tmp_path / "b", items=30, users=10).run()
        assert first["metrics"] == second["metrics"]

    def results(self, tmp_path: Path) -> None:
        config = small()
        result = Ablate(config=config, run_dir=tmp_path, items=30, users=10).run()

        payload = json.loads((tmp_path / "ablations.json").read_text(encoding="utf-8"))
        assert payload["cfg_hash"] == result["cfg_hash"]
        assert payload["metrics"] == result["metrics"]

        report = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "Ablation Report" in report
        for name in conditions(config):
            assert name in report

    def morel(self, tmp_path: Path) -> None:
        """Built-in ablations are available without registration."""
        assert "noretry" in ABLATIONS
        assert "nope" in ABLATIONS
        assert "nobook" in ABLATIONS
