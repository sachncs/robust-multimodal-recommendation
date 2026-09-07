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
from morel.core.log import get as logger
from morel.core.seed import seed as seed_everything

log = logger("data.cli")


def config(args: argparse.Namespace) -> Config:
    """Load the config named by ``--config``, or the defaults."""
    path = getattr(args, "config", None)
    return Config.load(path) if path else Config()


def paths(args: argparse.Namespace, cfg: Config) -> None:
    """Fill unset path, category and masking flags from ``cfg``.

    Every one of these had a hardcoded default that shadowed the
    corresponding cfg field, so configuring data.raw or data.category had
    no effect on any subcommand.
    """
    defaults = {
        "dest": cfg.data.raw,
        "data_dir": cfg.data.raw if args.cmd in {"extract", "build"} else cfg.data.processed,
        "out_dir": cfg.data.processed,
        "category": cfg.data.category,
        "min_edges": cfg.data.min,
        "ratio": cfg.masking.ratio,
        "kind": cfg.masking.kind,
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
    cfg = config(args)
    paths(args, cfg)
    configure_log(level=cfg.log.level, structured=cfg.log.structured)
    try:
        if args.cmd == "download":
            from morel.data.acquire import download

            downloaded = download(args.category, args.dest)
            for p in downloaded:
                print(p)
        elif args.cmd == "extract":
            run_extract(args, cfg)
        elif args.cmd == "build":
            assemble(args, cfg)
        elif args.cmd == "mask":
            from morel.data import build_mask

            seed_everything(cfg.seed)
            mask = build_mask(
                args.kind,
                items=args.items,
                modalities=args.modalities,
                ratio=args.ratio,
                seed=cfg.masking.seed,
            )
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            import numpy as np

            np.save(out, mask.numpy())
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


def run_extract(args: argparse.Namespace, cfg: Config) -> None:
    """Run the ``extract`` subcommand.

    The encoder for each modality is named by ``cfg.encoder.text`` and
    ``cfg.encoder.visual`` and built through the extractor registry, so a
    configured backbone is actually the one that runs. ``--synthetic`` forces
    the deterministic ``random`` encoder, which needs no model download.
    """
    from morel.data import assemble
    from morel.data.extract import text as encode_text
    from morel.data.manifest import Manifest
    from morel.data.store import store
    from morel.data.validate import features as check_features

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = max(8, cfg.encode.hidden // 8)
    dv = cfg.encoder.visual_dim
    dim_text = cfg.encoder.td
    text_kind = "random" if args.synthetic else cfg.encoder.text
    visual_kind = "random" if args.synthetic else cfg.encoder.visual
    log.info("extract.encoders", extra={"text": text_kind, "visual": visual_kind})

    text_encoder = assemble(text_kind, dim=dim_text, batch=cfg.encoder.batch, seed=cfg.seed + 1)
    visual_encoder = assemble(visual_kind, dim=dv, batch=cfg.encoder.batch, seed=cfg.seed)
    # Synthetic inputs are item ids; a real run would read them from data_dir.
    inputs = [f"item-{i}" for i in range(items)]
    feats = {
        "visual": encode_text(inputs, visual_encoder, batch=cfg.encoder.batch),
        "text": encode_text(inputs, text_encoder, batch=cfg.encoder.batch),
    }
    check_features(feats, items=items)
    # ``**feats`` is modality-keyed; a key colliding with store's own
    # keyword-only parameter raises rather than silently misbinding.
    store(
        out_dir / "features.npz",
        **feats,  # type: ignore[arg-type]
    )
    Manifest(
        dataset="synthetic" if args.synthetic else args.data_dir,
        version="0",
        code=f"morel.data.extract:{text_kind}+{visual_kind}",
        seed=cfg.seed,
        extractor="random",
        cfg_hash=cfg.hash(),
        extras={"items": items},
    )
    saved = (out_dir / "features.npz").resolve()
    sidecar = saved.with_suffix(saved.suffix + ".manifest.json")
    sidecar.write_text(
        Manifest(
            dataset="synthetic" if args.synthetic else args.data_dir,
            version="0",
            code=f"morel.data.extract:{text_kind}+{visual_kind}",
            seed=cfg.seed,
            extractor="random",
            cfg_hash=cfg.hash(),
            extras={"items": items},
        ).json(),
        encoding="utf-8",
    )
    print(out_dir / "features.npz")


def assemble(args: argparse.Namespace, cfg: Config) -> None:
    """Run the ``build`` subcommand."""
    from morel.data.build import bipartite as build_bipartite
    from morel.data.build import cooccurrence, kcore
    from morel.data.manifest import Manifest
    from morel.data.store import save_graph, store

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np

    rng = np.random.default_rng(cfg.seed)
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
    store(
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
            seed=cfg.seed,
            extractor="random",
            cfg_hash=cfg.hash(),
            extras={"users": users, "items": items, "min_edges": args.min_edges},
        ).json(),
        encoding="utf-8",
    )
    print(out_dir / "bipartite.npz")
    print(out_dir / "item_graph.npz")


if __name__ == "__main__":
    raise SystemExit(main())
