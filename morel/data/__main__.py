"""CLI entry point for ``python -m morel.data``.

Subcommands: download, extract, build, mask, verify.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from morel.core.config import Config
from morel.core.errors import MorelError
from morel.core.log import configure as configure_log
from morel.core.log import get as get_logger
from morel.core.seed import seed as seed_everything

log = get_logger("data.cli")


def main(argv: list[str] | None = None) -> int:
    """Dispatch subcommand."""
    parser = argparse.ArgumentParser(prog="morel.data", description="morel data lifecycle")
    sub = parser.add_subparsers(dest="cmd", required=True)

    download_cmd = sub.add_parser("download", help="download Amazon 5-core dataset")
    download_cmd.add_argument("--category", required=True)
    download_cmd.add_argument("--dest", default="data/raw")
    download_cmd.add_argument(
        "--legacy",
        action="store_true",
        help="use the legacy McAuley UCSD URL via download_legacy",
    )

    extract = sub.add_parser("extract", help="extract features from raw data")
    extract.add_argument("--data-dir", default="data/raw")
    extract.add_argument("--out-dir", default="data/processed")
    extract.add_argument("--config", default=None)
    extract.add_argument("--synthetic", action="store_true", help="use synthetic features")

    build = sub.add_parser("build", help="build bipartite and item graphs")
    build.add_argument("--data-dir", default="data/raw")
    build.add_argument("--out-dir", default="data/processed")
    build.add_argument("--config", default=None)
    build.add_argument("--synthetic", action="store_true", help="use synthetic interactions")

    mask_cmd = sub.add_parser("mask", help="generate modality mask")
    mask_cmd.add_argument("--items", type=int, required=True)
    mask_cmd.add_argument("--modalities", type=int, required=True)
    mask_cmd.add_argument("--ratio", type=float, default=0.4)
    mask_cmd.add_argument("--out", required=True)

    verify = sub.add_parser("verify", help="verify manifests under a directory")
    verify.add_argument("--data-dir", default="data/processed")

    args = parser.parse_args(argv)
    configure_log(level="INFO", structured=False)
    try:
        if args.cmd == "download":
            from morel.data.acquire import download, download_legacy

            download_fn = download_legacy if args.legacy else download
            paths = download_fn(args.category, args.dest)
            for p in paths:
                print(p)
        elif args.cmd == "extract":
            run_extract(args)
        elif args.cmd == "build":
            run_build(args)
        elif args.cmd == "mask":
            from morel.data.mask import bernoulli

            config = Config()
            seed_everything(config.seed)
            mask = bernoulli(args.items, args.modalities, args.ratio, seed=config.seed)
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            import numpy as np

            np.save(out, mask.to_numpy())
            print(out)
        elif args.cmd == "verify":
            for path in Path(args.data_dir).rglob("*.manifest.json"):
                print(path)
        else:
            parser.error(f"unknown subcommand {args.cmd!r}")
    except MorelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def run_extract(args: argparse.Namespace) -> None:
    """Run the ``extract`` subcommand."""
    from morel.core.config import Config as _Config
    from morel.data.extract import random as random_features
    from morel.data.manifest import Manifest
    from morel.data.store import save_npz
    from morel.data.validate import features as validate_features

    config = _Config.from_yaml(args.config) if args.config else _Config()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = max(8, config.encode.hidden // 8)
    dim_visual = config.encoder.visual_dim
    dim_text = config.encoder.text_dim
    feats = {
        "visual": random_features(items, dim_visual, seed=config.seed),
        "text": random_features(items, dim_text, seed=config.seed + 1),
    }
    validate_features(feats, items=items)
    # ``**feats`` is modality-keyed; a key colliding with save_npz's own
    # keyword-only parameter raises rather than silently misbinding.
    save_npz(
        out_dir / "features.npz",
        **feats,  # type: ignore[arg-type]
    )
    Manifest(
        dataset="synthetic" if args.synthetic else args.data_dir,
        version="0",
        code="morel.data.extract.random",
        seed=config.seed,
        extractor="random",
        config_hash=config.hash(),
        extras={"items": items},
    )
    saved = (out_dir / "features.npz").resolve()
    sidecar = saved.with_suffix(saved.suffix + ".manifest.json")
    sidecar.write_text(
        Manifest(
            dataset="synthetic" if args.synthetic else args.data_dir,
            version="0",
            code="morel.data.extract.random",
            seed=config.seed,
            extractor="random",
            config_hash=config.hash(),
            extras={"items": items},
        ).to_json(),
        encoding="utf-8",
    )
    print(out_dir / "features.npz")


def run_build(args: argparse.Namespace) -> None:
    """Run the ``build`` subcommand."""
    from morel.core.config import Config as _Config
    from morel.data.build import bipartite as build_bipartite
    from morel.data.build import item_cooccurrence
    from morel.data.manifest import Manifest
    from morel.data.store import save_graph, save_npz

    config = _Config.from_yaml(args.config) if args.config else _Config()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np

    rng = np.random.default_rng(config.seed)
    users, items = 32, 64
    pairs = rng.integers(0, users, size=512), rng.integers(0, items, size=512)
    ui = build_bipartite(pairs[0], pairs[1], users, items)
    save_graph(out_dir / "bipartite.npz", ui)
    item_adj = item_cooccurrence(ui)
    save_graph(out_dir / "item_graph.npz", item_adj)
    save_npz(
        out_dir / "bipartite_meta.npz",
        users=np.asarray([users], dtype=np.int64),
        items=np.asarray([items], dtype=np.int64),
    )
    sidecar = (out_dir / "bipartite.npz").with_suffix(".npz.manifest.json")
    sidecar.write_text(
        Manifest(
            dataset="synthetic" if args.synthetic else args.data_dir,
            version="0",
            code="morel.data.build.bipartite",
            seed=config.seed,
            extractor="random",
            config_hash=config.hash(),
            extras={"users": users, "items": items},
        ).to_json(),
        encoding="utf-8",
    )
    print(out_dir / "bipartite.npz")
    print(out_dir / "item_graph.npz")


if __name__ == "__main__":
    raise SystemExit(main())
