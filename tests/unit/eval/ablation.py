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

from morel.app import AblationExperiment
from morel.core.config import Config
from morel.core.errors import ConfigError
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
    return Config.from_dict(payload)


class Checker:
    """Aggregated test methods for this module."""

    def registered(self) -> None:
        assert set(ABLATIONS) == {"no_retrieval", "no_pe", "no_codebook"}

    def all(self) -> None:
        """A configured condition that is not registered would fail mid-sweep."""
        for name in Config().eval.ablations:
            assert name in ABLATIONS, f"{name} is named in eval.ablations but not registered"

    def baseline(self) -> None:
        config = small(eval={"ks": [5], "ablations": ["no_pe"]})
        assert conditions(config) == (BASELINE, "no_pe")

    def unchanged(self) -> None:
        config = small()
        assert ablate(config, BASELINE) is config

    def context(self) -> None:
        assert ablate(small(), "no_retrieval").retrieve.kind == "none"

    def encoding(self) -> None:
        assert ablate(small(), "no_pe").encode.pe == 0

    def quantization(self) -> None:
        assert ablate(small(), "no_codebook").codebook.kind == "identity"

    def the(self) -> None:
        config = small()
        before = replace(config)
        for name in ABLATIONS:
            ablate(config, name)
        assert config == before

    def component(self) -> None:
        config = small()
        ablated = ablate(config, "no_pe")
        assert ablated.codebook == config.codebook
        assert ablated.retrieve == config.retrieve
        assert ablated.recommend == config.recommend
        assert ablated.encode.pe != config.encode.pe

    def rejected(self) -> None:
        with pytest.raises(ConfigError, match="unknown ablation 'nope'; available: "):
            ablate(small(), "nope")

    def cutoff(self, tmp_path: Path) -> None:
        config = small(eval={"ks": [5, 10], "ablations": ["no_pe", "no_codebook"]})
        result = AblationExperiment(config=config, run_dir=tmp_path, items=30, users=10).run()

        assert set(result["metrics"]) == {"recall@5", "ndcg@5", "recall@10", "ndcg@10"}
        for values in result["metrics"].values():
            assert set(values) == {BASELINE, "no_pe", "no_codebook"}
            for value in values.values():
                assert 0.0 <= value <= 1.0

    def metrics(self, tmp_path: Path) -> None:
        """Regression: completion output never reached the ranker, so all conditions tied.

        A sweep where every condition scores identically measures nothing, and
        would have been reported as though it did.
        """
        config = small(eval={"ks": [5], "ablations": ["no_retrieval", "no_codebook"]})
        result = AblationExperiment(config=config, run_dir=tmp_path, items=30, users=10).run()

        recall = result["metrics"]["recall@5"]
        assert len({round(v, 6) for v in recall.values()}) > 1, (
            f"every condition scored the same: {recall}"
        )

    def reproducible(self, tmp_path: Path) -> None:
        config = small()
        first = AblationExperiment(config=config, run_dir=tmp_path / "a", items=30, users=10).run()
        second = AblationExperiment(config=config, run_dir=tmp_path / "b", items=30, users=10).run()
        assert first["metrics"] == second["metrics"]

    def results(self, tmp_path: Path) -> None:
        config = small()
        result = AblationExperiment(config=config, run_dir=tmp_path, items=30, users=10).run()

        payload = json.loads((tmp_path / "ablations.json").read_text(encoding="utf-8"))
        assert payload["config_hash"] == result["config_hash"]
        assert payload["metrics"] == result["metrics"]

        report = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "Ablation Report" in report
        for name in conditions(config):
            assert name in report

    def morel(self, tmp_path: Path) -> None:
        name = "shallow-test-only"

        @ABLATIONS.register(name)
        def shallow(config: Config) -> Config:
            return replace(config, recommend=replace(config.recommend, layers=0))

        try:
            assert ablate(small(), name).recommend.layers == 0
            config = small(eval={"ks": [5], "ablations": [name]})
            result = AblationExperiment(config=config, run_dir=tmp_path, items=30, users=10).run()
            assert name in result["metrics"]["recall@5"]
        finally:
            ABLATIONS.unregister(name)