"""Application services: experiment, benchmark, reproduce."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from morel.core.config import Config
from morel.core.fidelity import render_json, render_markdown
from morel.core.log import get as get_logger
from morel.core.seed import seed as seed_everything
from morel.core.types import Embedding
from morel.data.build import bipartite as build_bipartite
from morel.data.build import item_cooccurrence
from morel.data.mask import bernoulli
from morel.data.manifest import Manifest
from morel.pipeline import Pipeline
from morel.train.completion import Completion, CompletionConfig

log = get_logger("app.experiment")


def _synthetic_dataset(items: int, dim_visual: int, dim_text: int, users: int) -> dict:
    """Build a small reproducible synthetic dataset."""
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, users, size=items * 5), rng.integers(0, items, size=items * 5)
    ui = build_bipartite(pairs[0], pairs[1], users, items)
    adj = item_cooccurrence(ui)
    features = {
        "visual": rng.normal(size=(items, dim_visual)).astype(np.float32),
        "text": rng.normal(size=(items, dim_text)).astype(np.float32),
    }
    mask = bernoulli(items, 2, 0.4, seed=0).to_numpy()
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
    epochs: int = 1
    seed: int | None = None

    def run(self) -> dict:
        """Run a full experiment and write artifacts under ``run_dir``.

        Returns:
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
        self.config.to_yaml(self.run_dir / "config.yaml")
        start = time.time()

        dataset = _synthetic_dataset(self.items, self.dim_visual, self.dim_text, self.users)
        pipeline = Pipeline(
            self.config,
            dims={"visual": self.dim_visual, "text": self.dim_text},
        )
        pipeline.attach_corpus(dataset["features"], dataset["mask"], dataset["item_adj"])

        features_t = {k: Embedding(name=k, tensor=_to_torch(v)) for k, v in dataset["features"].items()}
        mask_t = _to_torch(dataset["mask"])
        from torch.utils.data import DataLoader, Dataset

        items_count = self.items

        class _Ds(Dataset):
            def __len__(self) -> int:
                return items_count

            def __getitem__(self, idx: int) -> dict:
                return {
                    "index": idx,
                    "features": {k: v[idx] for k, v in dataset["features"].items()},
                    "mask": dataset["mask"][idx],
                    "adjacency": dataset["item_adj"],
                }

        loader = DataLoader(_Ds(), batch_size=8, collate_fn=_collate)
        trainer = Completion(
            pipeline,
            CompletionConfig(),
            monitor=None,
            checkpoint_dir=self.run_dir,
        )
        history = trainer.fit(loader, None, epochs=self.epochs, patience=self.epochs + 1)

        metrics_path = self.run_dir / "metrics.jsonl"
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "time": datetime.now(tz=UTC).isoformat(),
                        "event": "experiment.end",
                        "duration_s": time.time() - start,
                        "best_loss": history.get("best", None),
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
                "epochs": self.epochs,
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
            f"- epochs: {self.epochs}\n"
            f"- best_loss: {history.get('best', None)}\n",
            encoding="utf-8",
        )

        render_markdown(self.run_dir / "FIDELITY.md")
        render_json(self.run_dir / "FIDELITY.json")

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
        }


def _to_torch(arr: np.ndarray):  # type: ignore[no-untyped-def]
    import torch

    return torch.from_numpy(arr)


def _collate(batch):  # type: ignore[no-untyped-def]
    import torch

    features_keys = list(batch[0]["features"].keys())
    return {
        "index": torch.from_numpy(np.stack([np.asarray(b["index"]) for b in batch])),
        "mask": torch.from_numpy(np.stack([np.asarray(b["mask"]) for b in batch])),
        "features": {
            k: torch.from_numpy(np.stack([b["features"][k] for b in batch]))
            for k in features_keys
        },
        "adjacency": batch[0]["adjacency"],
    }


@dataclass
class Benchmark:
    """Run a benchmark sweep and return timings."""

    config: Config
    run_dir: Path
    sizes: list[int] = field(default_factory=lambda: [16, 32])
    epochs: int = 1

    def run(self) -> dict:
        """Run benchmarks at the requested sizes."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, list[float]] = {}
        for size in self.sizes:
            dataset = _synthetic_dataset(size, 4, 2, max(2, size // 4))
            pipeline = Pipeline(
                self.config,
                dims={"visual": 4, "text": 2},
            )
            pipeline.attach_corpus(dataset["features"], dataset["mask"], dataset["item_adj"])
            features = {k: _to_torch(v) for k, v in dataset["features"].items()}
            mask = _to_torch(dataset["mask"])
            index_t = _to_torch(np.arange(size))
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

    def run(self) -> dict:
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


__all__ = ["Experiment", "Benchmark", "Reproduce"]
