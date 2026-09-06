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


def small_config(**overrides: Any) -> Config:
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


def test_shipped_ablations_are_registered() -> None:
    assert set(ABLATIONS.available()) == {"no_retrieval", "no_pe", "no_codebook"}


def test_config_default_ablations_are_all_registered() -> None:
    """A configured condition that is not registered would fail mid-sweep."""
    for name in Config().eval.ablations:
        assert name in ABLATIONS, f"{name} is named in eval.ablations but not registered"


def test_conditions_start_with_the_baseline() -> None:
    config = small_config(eval={"ks": [5], "ablations": ["no_pe"]})
    assert conditions(config) == (BASELINE, "no_pe")


def test_baseline_is_the_config_unchanged() -> None:
    config = small_config()
    assert ablate(config, BASELINE) is config


def test_no_retrieval_removes_graph_context() -> None:
    assert ablate(small_config(), "no_retrieval").retrieve.kind == "none"


def test_no_pe_removes_the_positional_encoding() -> None:
    assert ablate(small_config(), "no_pe").encode.pe == 0


def test_no_codebook_removes_quantization() -> None:
    assert ablate(small_config(), "no_codebook").codebook.kind == "identity"


def test_ablation_does_not_mutate_the_baseline() -> None:
    config = small_config()
    before = replace(config)
    for name in ABLATIONS.available():
        ablate(config, name)
    assert config == before


def test_ablation_changes_only_the_named_component() -> None:
    config = small_config()
    ablated = ablate(config, "no_pe")
    assert ablated.codebook == config.codebook
    assert ablated.retrieve == config.retrieve
    assert ablated.recommend == config.recommend
    assert ablated.encode.pe != config.encode.pe


def test_unknown_ablation_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown ablation 'nope'; available: "):
        ablate(small_config(), "nope")


def test_sweep_reports_every_condition_at_every_cutoff(tmp_path: Path) -> None:
    config = small_config(eval={"ks": [5, 10], "ablations": ["no_pe", "no_codebook"]})
    result = AblationExperiment(config=config, run_dir=tmp_path, items=30, users=10).run()

    assert set(result["metrics"]) == {"recall@5", "ndcg@5", "recall@10", "ndcg@10"}
    for values in result["metrics"].values():
        assert set(values) == {BASELINE, "no_pe", "no_codebook"}
        for value in values.values():
            assert 0.0 <= value <= 1.0


def test_ablations_actually_change_the_metrics(tmp_path: Path) -> None:
    """Regression: completion output never reached the ranker, so all conditions tied.

    A sweep where every condition scores identically measures nothing, and
    would have been reported as though it did.
    """
    config = small_config(eval={"ks": [5], "ablations": ["no_retrieval", "no_codebook"]})
    result = AblationExperiment(config=config, run_dir=tmp_path, items=30, users=10).run()

    recall = result["metrics"]["recall@5"]
    assert len({round(v, 6) for v in recall.values()}) > 1, (
        f"every condition scored the same: {recall}"
    )


def test_sweep_is_reproducible(tmp_path: Path) -> None:
    config = small_config()
    first = AblationExperiment(config=config, run_dir=tmp_path / "a", items=30, users=10).run()
    second = AblationExperiment(config=config, run_dir=tmp_path / "b", items=30, users=10).run()
    assert first["metrics"] == second["metrics"]


def test_sweep_writes_its_results(tmp_path: Path) -> None:
    config = small_config()
    result = AblationExperiment(config=config, run_dir=tmp_path, items=30, users=10).run()

    payload = json.loads((tmp_path / "ablations.json").read_text(encoding="utf-8"))
    assert payload["config_hash"] == result["config_hash"]
    assert payload["metrics"] == result["metrics"]

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Ablation Report" in report
    for name in conditions(config):
        assert name in report


def test_a_third_party_condition_needs_no_change_to_morel(tmp_path: Path) -> None:
    name = "shallow-test-only"

    @ABLATIONS.register(name)
    def shallow(config: Config) -> Config:
        return replace(config, recommend=replace(config.recommend, layers=0))

    try:
        assert ablate(small_config(), name).recommend.layers == 0
        config = small_config(eval={"ks": [5], "ablations": [name]})
        result = AblationExperiment(config=config, run_dir=tmp_path, items=30, users=10).run()
        assert name in result["metrics"]["recall@5"]
    finally:
        ABLATIONS.unregister(name)
