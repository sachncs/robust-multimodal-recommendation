"""The recommendation stage must actually be trainable end to end.

``morel train recommendation`` ran the completion experiment and printed
"recommendation trained", so ``morel.train.recommendation.Recommendation`` was
unreachable from any entry point. These tests pin that the recommendation
service exists, is what the CLI dispatches to, and honours its config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from morel.app import Rank
from morel.core.config import Config


def small(**recommendation: Any) -> Config:
    """Return a config small enough to train quickly."""
    return Config.parse(
        {
            "encode": {"hidden": 16, "pe": 4, "layers": 1, "heads": 2},
            "codebook": {"size": 16},
            "recommend": {"kind": "light", "embed": 8, "layers": 2},
            "recommendation": {"epochs": 2, "batch": 16, **recommendation},
        }
    )


class Checker:
    """Aggregated test methods for this module."""

    def run(self, tmp_path: Path) -> None:
        result = Rank(
            config=small(), run_dir=tmp_path, items=20, users=8
        ).run()

        for name in ("config.yaml", "manifest.json", "metrics.jsonl", "report.md"):
            assert (tmp_path / name).exists(), f"missing {name}"
        assert set(result) >= {"duration", "run_dir", "config_hash", "best", "train_loss"}

    def it(self, tmp_path: Path) -> None:
        Rank(config=small(), run_dir=tmp_path, items=20, users=8).run()

        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["code"] == "morel.app.Rank"
        assert manifest["extras"]["recommender"] == "light"

        events = [
            json.loads(line)["event"]
            for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert "recommendation.end" in events
        assert "experiment.end" not in events

    def reported(self, tmp_path: Path) -> None:
        """Regression: fit() left best at infinity whenever no val loader was given."""
        result = Rank(
            config=small(), run_dir=tmp_path, items=20, users=8
        ).run()
        assert 0.0 < float(result["best"]) < 100.0
        assert 0.0 < float(result["train_loss"]) < 100.0

    def no(self, tmp_path: Path) -> None:
        """With val=0 there is no held-out set, so the train loss is what is tracked."""
        result = Rank(
            config=small(val=0.0), run_dir=tmp_path, items=20, users=8
        ).run()
        assert result["train_loss"] == pytest.approx(result["best"])

    def epochs(self, tmp_path: Path) -> None:
        experiment = Rank(config=small(epochs=5), run_dir=tmp_path)
        assert experiment.resolved_epochs() == 5
        assert (
            Rank(
                config=small(epochs=5), run_dir=tmp_path, epochs=1
            ).resolved_epochs()
            == 1
        )

    def hyperparameters(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        import morel.app.experiment as module

        real = module.Recommendation

        def capture(model: Any, config: Any, **kwargs: Any) -> Any:
            captured["trainer_config"] = config
            captured.update(kwargs)
            return real(model, config, **kwargs)

        monkeypatch.setattr(module, "Recommendation", capture)
        config = small(lr=4e-4, weight_decay=2e-6, negatives=3, grad_clip=1.5)
        Rank(config=config, run_dir=tmp_path, items=20, users=8).run()

        assert captured["lr"] == pytest.approx(4e-4)
        assert captured["weight_decay"] == pytest.approx(2e-6)
        assert captured["negatives_count"] == 3
        assert captured["trainer_config"].grad_clip == pytest.approx(1.5)

    def recommender(self, tmp_path: Path) -> None:
        config = Config.parse(
            {
                "encode": {"hidden": 16, "pe": 4, "layers": 1, "heads": 2},
                "codebook": {"size": 16},
                "recommend": {"kind": "mf", "embed": 8, "layers": 2},
                "recommendation": {"epochs": 1, "batch": 16},
            }
        )
        Rank(config=config, run_dir=tmp_path, items=20, users=8).run()
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["extras"]["recommender"] == "mf"

    def cli(self, 
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The CLI must not quietly run the completion experiment instead."""
        monkeypatch.chdir(tmp_path)
        called: list[str] = []
        import morel.app.experiment as module

        real = module.Rank.run

        def spy(self: Any) -> Any:
            called.append("recommendation")
            self.epochs = 1
            return real(self)

        monkeypatch.setattr(module.Rank, "run", spy)

        from morel.cli import main

        assert main(["train", "recommendation"]) == 0
        assert called == ["recommendation"]