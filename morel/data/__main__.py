"""CLI entry point for ``python -m morel.data``.

Subcommands: download, extract, build, mask, verify.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from morel.core.config import Config
from morel.core.errors import Error
from morel.core.log import configure as configure_log
from morel.core.log import get as get_logger
from morel.core.seed import seed as seed_everything

log = get_logger("data.cli")


def load_config(args: argparse.Namespace) -> Config:
    """Load the config named by ``--config``, or the defaults."""
    path = getattr(args, "config", None)
    return Config.load(path) if path else Config()


def resolve_paths(args: argparse.Namespace, config: Config) -> None:
    """Fill unset path, category and masking flags from ``config``.

    Every one of these had a hardcoded default that shadowed the
    corresponding config field, so configuring data.raw or data.category had
    no effect on any subcommand.
    """
    defaults = {
        "dest": config.data.raw,
        "data_dir": config.data.raw if args.cmd in {"extract", "build"} else config.data.processed,
        "out_dir": config.data.processed,
        "category": config.data.category,
        "min_edges": config.data.min,
        "ratio": config.masking.ratio,
        "kind": config.masking.kind,
    }
    for name, value in defaults.items():
        if hasattr(args, name) and getattr(args, name) is None:
            setattr(args, name, value)


def main(argv: list[str] | None = None) -> int:
    """Dispatch subcommand."""
    parser = argparse.ArgumentParser(prog="morel.data", description="morel data lifecycle")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Path and category flags default to None so that the configuration
    # supplies them; an explicit flag still wins. Hardcoded defaults here
    # would silently shadow data.raw, data.processed and data.category.
    download_cmd = sub.add_parser("download", help="download Amazon 5-core dataset")
    download_cmd.add_argument("--category", default=None, help="overrides data.category")
    download_cmd.add_argument("--dest", default=None, help="overrides data.raw")
    download_cmd.add_argument("--config", default=None)

    extract = sub.add_parser("extract", help="extract features from raw data")
    extract.add_argument("--data-dir", default=None, help="overrides data.raw")
    extract.add_argument("--out-dir", default=None, help="overrides data.processed")
    extract.add_argument("--config", default=None)
    extract.add_argument("--synthetic", action="store_true", help="use synthetic features")

    build = sub.add_parser("build", help="build bipartite and item graphs")
    build.add_argument("--data-dir", default=None, help="overrides data.raw")
    build.add_argument("--out-dir", default=None, help="overrides data.processed")
    build.add_argument("--min-edges", type=int, default=None, help="overrides data.min")
    build.add_argument("--config", default=None)
    build.add_argument("--synthetic", action="store_true", help="use synthetic interactions")

    mask_cmd = sub.add_parser("mask", help="generate modality mask")
    mask_cmd.add_argument("--items", type=int, required=True)
    mask_cmd.add_argument("--modalities", type=int, required=True)
    mask_cmd.add_argument("--ratio", type=float, default=None, help="overrides masking.ratio")
    mask_cmd.add_argument("--kind", default=None, help="overrides masking.kind")
    mask_cmd.add_argument("--config", default=None)
    mask_cmd.add_argument("--out", required=True)

    verify = sub.add_parser("verify", help="verify manifests under a directory")
    verify.add_argument("--data-dir", default=None, help="overrides data.processed")
    verify.add_argument("--config", default=None)

    args = parser.parse_args(argv)
    config = load_config(args)
    resolve_paths(args, config)
    configure_log(level=config.log.level, structured=config.log.structured)
    try:
        if args.cmd == "download":
            from morel.data.acquire import download

            paths = download(args.category, args.dest)
            for p in paths:
                print(p)
        elif args.cmd == "extract":
            run_extract(args, config)
        elif args.cmd == "build":
            run_build(args, config)
        elif args.cmd == "mask":
            from morel.data import build_mask

            seed_everything(config.seed)
            mask = build_mask(
                args.kind,
                items=args.items,
                modalities=args.modalities,
                ratio=args.ratio,
                seed=config.masking.seed,
            )
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
    except Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def run_extract(args: argparse.Namespace, config: Config) -> None:
    """Run the ``extract`` subcommand.

    The encoder for each modality is named by ``config.encoder.text`` and
    ``config.encoder.visual`` and built through the extractor registry, so a
    configured backbone is actually the one that runs. ``--synthetic`` forces
    the deterministic ``random`` encoder, which needs no model download.
    """
    from morel.data import build_extractor
    from morel.data.extract import text as encode_text
    from morel.data.manifest import Manifest
    from morel.data.store import save_npz
    from morel.data.validate import features as check_features

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = max(8, config.encode.hidden // 8)
    dim_visual = config.encoder.visual_dim
    dim_text = config.encoder.text_dim
    text_kind = "random" if args.synthetic else config.encoder.text
    visual_kind = "random" if args.synthetic else config.encoder.visual
    log.info("extract.encoders", extra={"text": text_kind, "visual": visual_kind})

    text_encoder = build_extractor(
        text_kind, dim=dim_text, batch=config.encoder.batch, seed=config.seed + 1
    )
    visual_encoder = build_extractor(
        visual_kind, dim=dim_visual, batch=config.encoder.batch, seed=config.seed
    )
    # Synthetic inputs are item ids; a real run would read them from data_dir.
    inputs = [f"item-{i}" for i in range(items)]
    feats = {
        "visual": encode_text(inputs, visual_encoder, batch=config.encoder.batch),
        "text": encode_text(inputs, text_encoder, batch=config.encoder.batch),
    }
    check_features(feats, items=items)
    # ``**feats`` is modality-keyed; a key colliding with save_npz's own
    # keyword-only parameter raises rather than silently misbinding.
    save_npz(
        out_dir / "features.npz",
        **feats,  # type: ignore[arg-type]
    )
    Manifest(
        dataset="synthetic" if args.synthetic else args.data_dir,
        version="0",
        code=f"morel.data.extract:{text_kind}+{visual_kind}",
        seed=config.seed,
        extractor="random",
        cfg_hash=config.hash(),
        extras={"items": items},
    )
    saved = (out_dir / "features.npz").resolve()
    sidecar = saved.with_suffix(saved.suffix + ".manifest.json")
    sidecar.write_text(
        Manifest(
            dataset="synthetic" if args.synthetic else args.data_dir,
            version="0",
            code=f"morel.data.extract:{text_kind}+{visual_kind}",
            seed=config.seed,
            extractor="random",
            cfg_hash=config.hash(),
            extras={"items": items},
        ).to_json(),
        encoding="utf-8",
    )
    print(out_dir / "features.npz")


def run_build(args: argparse.Namespace, config: Config) -> None:
    """Run the ``build`` subcommand."""
    from morel.data.build import bipartite as build_bipartite
    from morel.data.build import cooccurrence, kcore
    from morel.data.manifest import Manifest
    from morel.data.store import save_graph, save_npz

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np

    rng = np.random.default_rng(config.seed)
    users, items = 32, 64
    pairs = rng.integers(0, users, size=512), rng.integers(0, items, size=512)
    ui = build_bipartite(pairs[0], pairs[1], users, items)
    save_graph(out_dir / "bipartite.npz", ui)
    item_adj = cooccurrence(ui)
    # data.min is the k-core threshold; leaving it unapplied meant the
    # configured minimum-degree filter never ran.
    if args.min_edges > 0:
        item_adj = kcore(item_adj, args.min_edges)
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
            cfg_hash=config.hash(),
            extras={"users": users, "items": items, "min_edges": args.min_edges},
        ).to_json(),
        encoding="utf-8",
    )
    print(out_dir / "bipartite.npz")
    print(out_dir / "item_graph.npz")


if __name__ == "__main__":
    raise SystemExit(main())
