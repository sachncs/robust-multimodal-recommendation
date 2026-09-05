"""Top-level CLI entry point for morel.

Dispatches to ``morel.data``, ``morel.train``, ``morel.eval``, ``morel.bench``,
``morel.reproduce``, and ``morel.serve`` subcommand handlers.

The top-level parser uses subparsers that consume the full remaining argv,
so each subcommand's parser sees only its own args.
"""

from __future__ import annotations

import argparse
import sys

from morel.core.log import configure as configure_log


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="morel", description="morel: graph retrieval-enhanced multimodal recommendation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("data", help="data lifecycle commands", add_help=False)
    sub.add_parser("train", help="training commands", add_help=False)
    sub.add_parser("eval", help="evaluation commands", add_help=False)
    sub.add_parser("bench", help="benchmark commands", add_help=False)
    sub.add_parser("reproduce", help="reproduction commands", add_help=False)
    sub.add_parser("serve", help="inference server", add_help=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch to subcommands."""
    configure_log(level="INFO", structured=False)
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    if not raw:
        parser.print_help()
        return 0
    cmd = raw[0]
    rest = raw[1:]
    if cmd not in {"data", "train", "eval", "bench", "reproduce", "serve"}:
        parser.error(f"unknown command {cmd!r}")
        return 2
    handler = {
        "data": _run_data,
        "train": _run_train,
        "eval": _run_eval,
        "bench": _run_bench,
        "reproduce": _run_reproduce,
        "serve": _run_serve,
    }[cmd]
    return handler(rest)


def _run_data(argv: list[str]) -> int:
    from morel.data.__main__ import main as data_main

    return int(data_main(argv) or 0)


def _run_train(argv: list[str]) -> int:
    print("morel train: command registered, full implementation lands in Phase 12", file=sys.stderr)
    return 0


def _run_eval(argv: list[str]) -> int:
    print("morel eval: command registered, full implementation lands in Phase 12", file=sys.stderr)
    return 0


def _run_bench(argv: list[str]) -> int:
    print("morel bench: command registered, full implementation lands in Phase 12", file=sys.stderr)
    return 0


def _run_reproduce(argv: list[str]) -> int:
    print("morel reproduce: command registered, full implementation lands in Phase 12", file=sys.stderr)
    return 0


def _run_serve(argv: list[str]) -> int:
    return _serve(argv)


def _serve(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="morel serve", description="inference server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    try:
        import uvicorn

        from morel.serve.app import create
    except ImportError:
        print("morel serve requires the [serve] extra: pip install morel[serve]", file=sys.stderr)
        return 1
    app = create()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
