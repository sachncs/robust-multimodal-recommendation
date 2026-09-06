"""The data CLI must read the configuration rather than hardcoding defaults.

Every path, the category and the masking settings had a hardcoded default in
the argument parser, which shadowed ``data.raw``, ``data.processed``,
``data.category``, ``data.min`` and ``masking.*``. Configuring them had no
effect on any subcommand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from morel.core.config import Config
from morel.data.__main__ import main


def write(tmp_path: Path, **payload: Any) -> Path:
    """Write a config YAML and return its path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.yaml"
    Config.from_dict(payload).to_yaml(path)
    return path


class Checker:
    """Aggregated test methods for this module."""

    def build(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        path = write(tmp_path, data={"processed": "custom-out"})
        assert main(["build", "--config", str(path)]) == 0
        assert (tmp_path / "custom-out" / "bipartite.npz").exists()

    def applies(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """data.min is the k-core threshold and was never applied."""
        monkeypatch.chdir(tmp_path)
        path = write(tmp_path, data={"processed": "out", "min": 4})
        assert main(["build", "--config", str(path)]) == 0

        manifest = json.loads(
            (tmp_path / "out" / "bipartite.npz.manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["extras"]["min_edges"] == 4

    def kcore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from morel.data.store import load_graph

        monkeypatch.chdir(tmp_path)
        loose = write(tmp_path, data={"processed": "loose", "min": 0})
        # 30 is high enough to bite on this synthetic graph, whose minimum item
        # degree is 21; a small threshold would remove nothing and the assertion
        # below would be vacuous.
        strict = write(tmp_path / "s", data={"processed": "strict", "min": 30})
        assert main(["build", "--config", str(loose)]) == 0
        assert main(["build", "--config", str(strict)]) == 0

        unfiltered = load_graph(tmp_path / "loose" / "item_graph.npz")
        filtered = load_graph(tmp_path / "strict" / "item_graph.npz")
        assert filtered.nnz < unfiltered.nnz, "a higher k-core must remove edges"

    def explicit(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        path = write(tmp_path, data={"processed": "from-config"})
        assert main(["build", "--config", str(path), "--out-dir", "from-flag"]) == 0
        assert (tmp_path / "from-flag" / "bipartite.npz").exists()
        assert not (tmp_path / "from-config").exists()

    def mask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        sparse = write(tmp_path, masking={"ratio": 0.1})
        assert (
            main(
                [
                    "mask",
                    "--items",
                    "200",
                    "--modalities",
                    "3",
                    "--out",
                    "a.npy",
                    "--config",
                    str(sparse),
                ]
            )
            == 0
        )
        dense = write(tmp_path / "d", masking={"ratio": 0.9})
        assert (
            main(
                [
                    "mask",
                    "--items",
                    "200",
                    "--modalities",
                    "3",
                    "--out",
                    "b.npy",
                    "--config",
                    str(dense),
                ]
            )
            == 0
        )

        assert np.load(tmp_path / "a.npy").mean() > np.load(tmp_path / "b.npy").mean()

    def uses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        one = write(tmp_path, masking={"seed": 1})
        two = write(tmp_path / "t", masking={"seed": 2})
        assert (
            main(["mask", "--items", "50", "--modalities", "2", "--out", "a.npy", "--config", str(one)])
            == 0
        )
        assert (
            main(["mask", "--items", "50", "--modalities", "2", "--out", "b.npy", "--config", str(two)])
            == 0
        )
        assert not np.array_equal(np.load(tmp_path / "a.npy"), np.load(tmp_path / "b.npy"))

    def kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        path = write(tmp_path, masking={"kind": "block", "ratio": 0.5})
        assert (
            main(
                ["mask", "--items", "40", "--modalities", "3", "--out", "m.npy", "--config", str(path)]
            )
            == 0
        )
        mask = np.load(tmp_path / "m.npy")
        assert (mask.sum(axis=1) > 0).all()

    def download(tmp_path: Path) -> None:
        """--category used to be required, ignoring data.category entirely."""
        import argparse

        from morel.data.__main__ import load_config, resolve_paths

        config = Config.from_dict({"data": {"category": "Books", "raw": "r", "processed": "p"}})
        args = argparse.Namespace(cmd="download", category=None, dest=None, config=None)
        resolve_paths(args, config)
        assert args.category == "Books"
        assert args.dest == "r"
        assert load_config(argparse.Namespace(config=None)) == Config()
