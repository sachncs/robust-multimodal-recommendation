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

    download = sub.add_parser("download", help="download Amazon 5-core dataset")
    download.add_argument("--category", required=True)
    download.add_argument("--dest", default="data/raw")

    extract = sub.add_parser("extract", help="extract features from raw data")
    extract.add_argument("--data-dir", default="data/raw")
    extract.add_argument("--out-dir", default="data/processed")
    extract.add_argument("--config", default=None)

    build = sub.add_parser("build", help="build bipartite and item graphs")
    build.add_argument("--data-dir", default="data/raw")
    build.add_argument("--out-dir", default="data/processed")
    build.add_argument("--config", default=None)

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
            from morel.data.acquire import download

            paths = download(args.category, args.dest)
            for p in paths:
                print(p)
        elif args.cmd == "extract":
            print(f"extract from {args.data_dir} -> {args.out_dir} (synthetic)")
        elif args.cmd == "build":
            print(f"build from {args.data_dir} -> {args.out_dir} (synthetic)")
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


if __name__ == "__main__":
    raise SystemExit(main())
