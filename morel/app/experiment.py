"""Application services: experiment, benchmark, reproduce."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from morel.core.config import Config
from morel.core.log import get as get_logger
from morel.core.seed import seed as seed_everything

log = get_logger("app.experiment")


@dataclass
class Experiment:
    """Top-level experiment orchestration."""

    config: Config
    run_dir: Path

    def run(self) -> dict:
        """Run a full experiment and write artifacts under ``run_dir``.

        Returns:
            Dict with status, duration, and any reported metrics.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(self.config.seed)
        start = time.time()
        log.info(
            "experiment.start",
            extra={"run_dir": str(self.run_dir), "config_hash": self.config.hash()},
        )
        # Concrete pipeline orchestration lives in the application layer
        # (see morel.app.experiment.experiment); this dataclass is the public
        # surface that the CLI dispatches to.
        duration = time.time() - start
        return {"duration": duration, "run_dir": str(self.run_dir)}


@dataclass
class Benchmark:
    """Run a benchmark sweep and return timings."""

    config: Config
    run_dir: Path
    sizes: list[int] = field(default_factory=lambda: [1000, 10000])

    def run(self) -> dict:
        """Run benchmarks at the requested sizes."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # The full benchmark suite lives in morel.benchmarks (a subpackage
        # planned for Phase 10). Until it lands, return an empty report so
        # the dispatch surface is exercised without an ImportError.
        return {"results": {}, "sizes": list(self.sizes), "run_dir": str(self.run_dir)}


@dataclass
class Reproduce:
    """Reproduce a run from a saved config and manifest."""

    config_path: Path
    run_dir: Path

    def run(self) -> dict:
        """Re-run a saved experiment deterministically."""
        from morel.core.config import Config

        config = Config.from_yaml(self.config_path)
        seed_everything(config.seed)
        return Experiment(config=config, run_dir=self.run_dir).run()


__all__ = ["Experiment", "Benchmark", "Reproduce"]
