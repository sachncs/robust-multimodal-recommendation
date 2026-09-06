"""Application services: experiment, benchmark, reproduce."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

from morel.app.data import (
    build_completion_loaders,
    build_recommendation_loaders,
    numpy_to_tensor,
    synth_bipartite,
)
from morel.core.config import Config, Masking
from morel.core.fidelity import render, render_md
from morel.core.log import get as get_logger
from morel.core.seed import seed as seed_everything
from morel.data import build_mask
from morel.data.build import bipartite as build_bipartite
from morel.data.build import cooccurrence
from morel.data.manifest import Manifest
from morel.eval import ablate, ablation_results, conditions, ndcg_at_k, recall_at_k
from morel.pipeline import Pipeline
from morel.train.completion import Completion, CompletionConfig
from morel.train.recommendation import Recommendation, RecommendationConfig

log = get_logger("app.experiment")


def synthetic(
    items: int,
    dim_visual: int,
    dim_text: int,
    users: int,
    masking: Masking | None = None,
) -> dict[str, Any]:
    """Build a small reproducible synthetic dataset.

    Args:
        items: Number of items.
        dim_visual: Visual feature width.
        dim_text: Text feature width.
        users: Number of users.
        masking: Masking settings; defaults to the shipped configuration. The
            missing-modality pattern is the experimental condition, so leaving
            it hardcoded meant a configured ratio had no effect on the run.

    Returns
    -------
        Dict with the interaction matrix, item graph, features and mask.
    """
    settings = masking if masking is not None else Masking()
    rng = np.random.default_rng(0)
    user_ids, item_ids = synth_bipartite(rng, items=items, users=users)
    ui = build_bipartite(user_ids, item_ids, users, items)
    adj = cooccurrence(ui)
    features = {
        "visual": rng.normal(size=(items, dim_visual)).astype(np.float32),
        "text": rng.normal(size=(items, dim_text)).astype(np.float32),
    }
    mask = build_mask(
        settings.kind,
        items=items,
        modalities=2,
        ratio=settings.ratio,
        seed=settings.seed,
    ).to_numpy()
    return {
        "ui": ui,
        "item_adj": adj,
        "features": features,
        "mask": mask,
        "users": users,
    }


@dataclass
class Experiment:
    """Top-level experiment orchestration."""

    config: Config
    run_dir: Path
    items: int = 50
    users: int = 20
    dim_visual: int = 8
    dim_text: int = 4
    epochs: int | None = None
    seed: int | None = None

    def resolved_epochs(self) -> int:
        """Return the epoch count to train for.

        An explicit ``epochs`` on the experiment wins; otherwise the value
        comes from ``config.completion.epochs``. Reading it from the config by
        default matters because the config is what gets hashed into the run
        manifest: a run that trained for a different number of epochs than its
        recorded config would not be reproducible from that record.
        """
        if self.epochs is not None:
            return self.epochs
        return self.config.completion.epochs

    def run(self) -> dict[str, Any]:
        """Run a full experiment and write artifacts under ``run_dir``.

        Returns
        -------
            Dict with status, duration, metrics, and config_hash.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        seed_value = self.seed if self.seed is not None else self.config.seed
        seed_everything(seed_value)
        config_hash = self.config.hash()
        log.info(
            "experiment.start",
            extra={"run_dir": str(self.run_dir), "config_hash": config_hash},
        )
        self.config.save(self.run_dir / "config.yaml")
        start = time.time()

        dataset = synthetic(
            self.items, self.dim_visual, self.dim_text, self.users, self.config.masking
        )
        pipeline = Pipeline(
            self.config,
            dims={"visual": self.dim_visual, "text": self.dim_text},
        )
        pipeline.attach(dataset["features"], dataset["mask"], dataset["item_adj"])

        loader, val_loader = build_completion_loaders(
            dataset["features"],
            dataset["mask"],
            dataset["item_adj"],
            batch_size=self.config.completion.batch,
            val_fraction=self.config.completion.val,
            seed=seed_value,
        )
        trainer = Completion(
            pipeline,
            CompletionConfig(
                lambda_usage=self.config.completion.usage,
                lambda_balance=self.config.completion.balance,
                grad_clip=self.config.completion.grad_clip,
            ),
            lr=self.config.completion.lr,
            weight_decay=self.config.completion.weight_decay,
            amp=self.config.completion.amp,
            device=self.config.device,
            monitor=None,
            checkpoint_dir=self.run_dir,
        )
        epochs = self.resolved_epochs()
        history = trainer.fit(
            loader, val_loader, epochs=epochs, patience=self.config.completion.patience
        )

        metrics_path = self.run_dir / "metrics.jsonl"
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "time": datetime.now(tz=UTC).isoformat(),
                        "event": "experiment.end",
                        "duration_s": time.time() - start,
                        "best_loss": history.get("best", None),
                        "train_loss": history.get("train_loss", None),
                        "config_hash": config_hash,
                    }
                )
                + "\n"
            )

        manifest = Manifest(
            dataset="synthetic",
            version="0",
            code="morel.experiment",
            seed=seed_value,
            extractor="synthetic",
            config_hash=config_hash,
            parents=[],
            extras={
                "items": self.items,
                "users": self.users,
                "epochs": epochs,
            },
        )
        manifest_path = self.run_dir / "manifest.json"
        manifest_path.write_text(manifest.to_json(), encoding="utf-8")

        report = self.run_dir / "report.md"
        report.write_text(
            "# morel — Experiment Report\n\n"
            f"- run_dir: `{self.run_dir}`\n"
            f"- config_hash: `{config_hash}`\n"
            f"- items: {self.items}\n"
            f"- users: {self.users}\n"
            f"- epochs: {epochs}\n"
            f"- best_loss: {history.get('best', None)}\n"
            f"- train_loss: {history.get('train_loss', None)}\n",
            encoding="utf-8",
        )

        render_md(self.run_dir / "FIDELITY.md")
        render(self.run_dir / "FIDELITY.json")

        duration = time.time() - start
        log.info(
            "experiment.end",
            extra={"duration": duration, "best_loss": history.get("best", None)},
        )
        return {
            "duration": duration,
            "run_dir": str(self.run_dir),
            "config_hash": config_hash,
            "best": history.get("best", None),
            "train_loss": history.get("train_loss", None),
        }


@dataclass
class RecommendationExperiment:
    """Train the downstream ranker with BPR and write artifacts.

    ``morel train recommendation`` previously ran the completion experiment
    and reported it as recommendation training, so the ``Recommendation``
    trainer was unreachable from any entry point. This is the service that
    actually drives it.
    """

    config: Config
    run_dir: Path
    items: int = 50
    users: int = 20
    epochs: int | None = None
    seed: int | None = None

    def resolved_epochs(self) -> int:
        """Return the epoch count, falling back to ``config.recommendation.epochs``."""
        if self.epochs is not None:
            return self.epochs
        return self.config.recommendation.epochs

    def run(self) -> dict[str, Any]:
        """Train the ranker and write config, manifest, metrics and report.

        Returns
        -------
            Dict with duration, run_dir, config_hash and best validation loss.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        seed_value = self.seed if self.seed is not None else self.config.seed
        seed_everything(seed_value)
        config_hash = self.config.hash()
        log.info(
            "recommendation.start",
            extra={"run_dir": str(self.run_dir), "config_hash": config_hash},
        )
        self.config.save(self.run_dir / "config.yaml")
        start = time.time()

        dataset = synthetic(self.items, 8, 4, self.users, self.config.masking)
        ui = dataset["ui"]
        pipeline = Pipeline(self.config, dims={"visual": 8, "text": 4})
        pipeline.attach_recommend(ui)
        assert pipeline.recommender is not None

        loader, val_loader = build_recommendation_loaders(
            ui,
            batch_size=self.config.recommendation.batch,
            val_fraction=self.config.recommendation.val,
            seed=seed_value,
        )
        trainer = Recommendation(
            pipeline.recommender,
            RecommendationConfig(grad_clip=self.config.recommendation.grad_clip),
            ui_graph=ui,
            negatives_count=self.config.recommendation.negatives,
            seed=seed_value,
            lr=self.config.recommendation.lr,
            weight_decay=self.config.recommendation.weight_decay,
            amp=self.config.recommendation.amp,
            device=self.config.device,
            monitor=None,
            checkpoint_dir=self.run_dir,
        )
        epochs = self.resolved_epochs()
        history = trainer.fit(
            loader, val_loader, epochs=epochs, patience=self.config.recommendation.patience
        )

        metrics_path = self.run_dir / "metrics.jsonl"
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "time": datetime.now(tz=UTC).isoformat(),
                        "event": "recommendation.end",
                        "duration_s": time.time() - start,
                        "best_loss": history.get("best", None),
                        "train_loss": history.get("train_loss", None),
                        "config_hash": config_hash,
                    }
                )
                + "\n"
            )

        manifest = Manifest(
            dataset="synthetic",
            version="0",
            code="morel.app.RecommendationExperiment",
            seed=seed_value,
            extractor="synthetic",
            config_hash=config_hash,
            parents=[],
            extras={
                "items": self.items,
                "users": self.users,
                "epochs": epochs,
                "recommender": self.config.recommend.kind,
            },
        )
        (self.run_dir / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")
        (self.run_dir / "report.md").write_text(
            "# morel — Recommendation Report\n\n"
            f"- run_dir: `{self.run_dir}`\n"
            f"- config_hash: `{config_hash}`\n"
            f"- recommender: {self.config.recommend.kind}\n"
            f"- items: {self.items}\n"
            f"- users: {self.users}\n"
            f"- epochs: {epochs}\n"
            f"- best_loss: {history.get('best', None)}\n"
            f"- train_loss: {history.get('train_loss', None)}\n",
            encoding="utf-8",
        )

        duration = time.time() - start
        log.info(
            "recommendation.end",
            extra={"duration": duration, "best_loss": history.get("best", None)},
        )
        return {
            "duration": duration,
            "run_dir": str(self.run_dir),
            "config_hash": config_hash,
            "best": history.get("best", None),
            "train_loss": history.get("train_loss", None),
        }


@dataclass
class AblationExperiment:
    """Run every condition in ``config.eval.ablations`` and compare them.

    ``eval.ablations`` listed condition names that nothing could act on. This
    runs the baseline and each named ablation through the same pipeline,
    changing only the component the condition removes, and reports the ranking
    metrics at every cutoff in ``eval.ks``.
    """

    config: Config
    run_dir: Path
    items: int = 50
    users: int = 20
    dim_visual: int = 8
    dim_text: int = 4
    seed: int | None = None

    def score(self, config: Config, dataset: dict[str, Any]) -> np.ndarray:
        """Build a pipeline under ``config`` and return its ``(users, items)`` scores.

        The completion output is fed into the ranker. Without that the ranker
        depends only on the interaction graph, every completion-stage ablation
        scores identically, and the sweep measures nothing.
        """
        feature_dim = self.dim_visual + self.dim_text
        pipeline = Pipeline(config, dims={"visual": self.dim_visual, "text": self.dim_text})
        pipeline.attach(dataset["features"], dataset["mask"], dataset["item_adj"])
        pipeline.attach_recommend(dataset["ui"], feature_dim=feature_dim)
        assert pipeline.recommender is not None
        index = torch.arange(self.items)
        output = pipeline(
            {name: torch.from_numpy(value) for name, value in dataset["features"].items()},
            torch.from_numpy(dataset["mask"]),
            dataset["item_adj"],
            index=index,
            training=False,
        )
        completed = torch.cat([output.completed[name] for name in ("visual", "text")], dim=-1)
        scores = pipeline.recommender(
            torch.arange(self.users), index, dataset["ui"], item_features=completed
        )
        return np.asarray(scores.detach().cpu().numpy())

    def run(self) -> dict[str, Any]:
        """Run the sweep and write artifacts under ``run_dir``.

        Returns
        -------
            Dict with the config hash and a metric-per-condition mapping.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        seed_value = self.seed if self.seed is not None else self.config.seed
        config_hash = self.config.hash()
        self.config.save(self.run_dir / "config.yaml")
        start = time.time()
        log.info("ablation.start", extra={"conditions": list(conditions(self.config))})

        dataset = synthetic(
            self.items, self.dim_visual, self.dim_text, self.users, self.config.masking
        )
        labels = dataset["ui"].sign().toarray()

        scores_by_condition: dict[str, np.ndarray] = {}
        for name in conditions(self.config):
            # Reseed per condition so each is scored from the same starting
            # state; otherwise a condition's numbers would depend on how many
            # conditions ran before it.
            seed_everything(seed_value)
            scores_by_condition[name] = self.score(ablate(self.config, name), dataset)

        metrics: dict[str, dict[str, float]] = {}
        for k in self.config.eval.ks:
            metrics[f"recall@{k}"] = ablation_results(
                scores_by_condition, labels, metric=partial(recall_at_k, k=k)
            )
            metrics[f"ndcg@{k}"] = ablation_results(
                scores_by_condition, labels, metric=partial(ndcg_at_k, k=k)
            )

        (self.run_dir / "ablations.json").write_text(
            json.dumps({"config_hash": config_hash, "metrics": metrics}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        rows = "\n".join(
            f"| {name} | " + " | ".join(f"{metrics[m][name]:.4f}" for m in sorted(metrics)) + " |"
            for name in conditions(self.config)
        )
        header = " | ".join(sorted(metrics))
        divider = " | ".join("---" for _ in sorted(metrics))
        (self.run_dir / "report.md").write_text(
            "# morel — Ablation Report\n\n"
            f"- config_hash: `{config_hash}`\n"
            f"- seed: {seed_value}\n\n"
            f"| condition | {header} |\n| --- | {divider} |\n{rows}\n",
            encoding="utf-8",
        )

        duration = time.time() - start
        log.info("ablation.end", extra={"duration": duration})
        return {
            "duration": duration,
            "run_dir": str(self.run_dir),
            "config_hash": config_hash,
            "metrics": metrics,
        }


@dataclass
class Benchmark:
    """Run a benchmark sweep and return timings."""

    config: Config
    run_dir: Path
    sizes: list[int] = field(default_factory=lambda: [16, 32])
    epochs: int = 1

    def run(self) -> dict[str, Any]:
        """Run benchmarks at the requested sizes."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, list[float]] = {}
        for size in self.sizes:
            dataset = synthetic(size, 4, 2, max(2, size // 4))
            pipeline = Pipeline(
                self.config,
                dims={"visual": 4, "text": 2},
            )
            pipeline.attach(dataset["features"], dataset["mask"], dataset["item_adj"])
            features = {k: numpy_to_tensor(v) for k, v in dataset["features"].items()}
            mask = numpy_to_tensor(dataset["mask"])
            index_t = numpy_to_tensor(np.arange(size))
            start = time.time()
            for _ in range(self.epochs):
                pipeline(features, mask, dataset["item_adj"], index=index_t, training=True)
            elapsed = time.time() - start
            results.setdefault("forward_s", []).append(elapsed)
        return {"results": results, "sizes": list(self.sizes), "run_dir": str(self.run_dir)}


@dataclass
class Reproduce:
    """Reproduce a run from a saved config and manifest."""

    config_path: Path
    run_dir: Path
    items: int = 50
    users: int = 20
    epochs: int = 1

    def run(self) -> dict[str, Any]:
        """Re-run a saved experiment deterministically."""
        config = Config.from_yaml(self.config_path)
        seed_everything(config.seed)
        experiment = Experiment(
            config=config,
            run_dir=self.run_dir,
            items=self.items,
            users=self.users,
            epochs=self.epochs,
            seed=config.seed,
        )
        return experiment.run()


__all__ = ["Benchmark", "Experiment", "Reproduce"]
