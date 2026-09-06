"""The experiment must actually use the configuration it records.

``Experiment.run`` wrote ``config.yaml`` into the run directory and hashed the
same config into ``manifest.json``, but built its trainer from
``FitConfig()`` defaults and a hardcoded batch size and epoch count.
Every ``completion.*`` hyperparameter was therefore ignored while still being
recorded, so the manifest described a run that had not happened. These tests
pin that the recorded config is the config that ran.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from morel.app.experiment import Experiment
from morel.core.config import Config


def small(**completion: Any) -> Config:
    """Return a config small enough to train quickly, with overrides applied."""
    payload: dict[str, Any] = {
        "encode": {"hidden": 16, "pe": 4, "layers": 1, "heads": 2},
        "codebook": {"size": 16},
        "completion": {"epochs": 2, "batch": 4, **completion},
    }
    return Config.parse(payload)


class Checker:
    """Aggregated test methods for this module."""

    def resolved(self) -> None:
        config = small(epochs=7)
        experiment = Experiment(config=config, run_dir=Path("unused"))
        assert experiment.resolved_epochs() == 7

    def explicit(self) -> None:
        config = small(epochs=7)
        experiment = Experiment(config=config, run_dir=Path("unused"), epochs=1)
        assert experiment.resolved_epochs() == 1

    def completion(self, 
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: these were dropped on the floor and defaults used instead."""
        captured: dict[str, Any] = {}
        import morel.app.experiment as module

        real = module.Completion

        def capture(model: Any, config: Any, **kwargs: Any) -> Any:
            captured["completion_config"] = config
            captured.update(kwargs)
            return real(model, config, **kwargs)

        monkeypatch.setattr(module, "Completion", capture)

        config = small(
            lr=5e-4,
            weight_decay=3e-6,
            usage=0.25,
            balance=0.75,
            grad_clip=2.5,
        )
        Experiment(config=config, run_dir=tmp_path, items=10, users=4).run()

        assert captured["lr"] == pytest.approx(5e-4)
        assert captured["weight_decay"] == pytest.approx(3e-6)
        assert captured["completion_config"].lambda_usage == pytest.approx(0.25)
        assert captured["completion_config"].lambda_balance == pytest.approx(0.75)
        assert captured["completion_config"].grad_clip == pytest.approx(2.5)

    def batch(self, 
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        import morel.app.experiment as module

        real = module.build_loaders

        def capture(*args: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(module, "build_loaders", capture)

        Experiment(config=small(batch=3), run_dir=tmp_path, items=10, users=4).run()
        assert captured["batch_size"] == 3

    def manifest(self, tmp_path: Path) -> None:
        config = small(epochs=3)
        Experiment(config=config, run_dir=tmp_path, items=10, users=4).run()

        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["extras"]["epochs"] == 3
        assert "- epochs: 3" in (tmp_path / "report.md").read_text(encoding="utf-8")

    def written(self, tmp_path: Path) -> None:
        """The config written beside a run must be the one its hash was taken from."""
        config = small(lr=7e-4, epochs=2)
        result = Experiment(config=config, run_dir=tmp_path, items=10, users=4).run()

        written = Config.load(tmp_path / "config.yaml")
        assert written.hash() == result["config_hash"]
        assert written.completion.lr == pytest.approx(7e-4)