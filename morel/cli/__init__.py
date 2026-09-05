"""Top-level CLI entry point for morel.

Dispatches to ``morel.data``, ``morel.train``, ``morel.eval``, ``morel.bench``,
``morel.reproduce``, ``morel.serve``, and ``morel.render-fidelity`` subcommands.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from morel.core.log import configure as configure_log
from morel.core.log import get as get_logger

log = get_logger("cli")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="morel", description="morel: graph retrieval-enhanced multimodal recommendation"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("data", help="data lifecycle commands", add_help=False)
    sub.add_parser("train", help="training commands", add_help=False)
    sub.add_parser("eval", help="evaluation commands", add_help=False)
    sub.add_parser("bench", help="benchmark commands", add_help=False)
    sub.add_parser("reproduce", help="reproduction commands", add_help=False)
    sub.add_parser("serve", help="inference server", add_help=False)
    sub.add_parser(
        "render-fidelity",
        help="render the fidelity registry as markdown/json",
        add_help=False,
    )
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
    if cmd not in {
        "data",
        "train",
        "eval",
        "bench",
        "reproduce",
        "serve",
        "render-fidelity",
    }:
        parser.error(f"unknown command {cmd!r}")
        return 2
    handler = {
        "data": _run_data,
        "train": _run_train,
        "eval": _run_eval,
        "bench": _run_bench,
        "reproduce": _run_reproduce,
        "serve": _run_serve,
        "render-fidelity": _run_render_fidelity,
    }[cmd]
    return handler(rest)


def _run_data(argv: list[str]) -> int:
    from morel.data.__main__ import main as data_main

    return int(data_main(argv) or 0)


def _run_train(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="morel train", description="training")
    sub = parser.add_subparsers(dest="sub", required=True)
    sub.add_parser("completion", help="train the completion stage", add_help=False)
    sub.add_parser("recommendation", help="train the recommendation stage", add_help=False)
    args = parser.parse_args(argv)

    config_path = _resolve_config_path(argv)
    if args.sub == "completion":
        from morel.app import Experiment

        config = _load_config_or_default(config_path)
        run_dir = Path("runs") / "completion"
        exp = Experiment(config=config, run_dir=run_dir, items=50, users=20, epochs=1)
        result = exp.run()
        print(f"completion trained: {result}")
        return 0
    if args.sub == "recommendation":
        from morel.app import Experiment

        config = _load_config_or_default(config_path)
        run_dir = Path("runs") / "recommendation"
        exp = Experiment(config=config, run_dir=run_dir, items=50, users=20, epochs=1)
        result = exp.run()
        print(f"recommendation trained: {result}")
        return 0
    parser.error(f"unknown train subcommand {args.sub!r}")
    return 2


def _run_eval(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="morel eval", description="evaluation")
    sub = parser.add_subparsers(dest="sub", required=True)
    sub.add_parser("rank", help="rank evaluation", add_help=False)
    sub.add_parser("robustness", help="robustness evaluation", add_help=False)
    args = parser.parse_args(argv)
    log.info("eval.start", extra={"sub": args.sub})
    if args.sub == "rank":
        from morel.eval import ndcg_at_k, recall_at_k

        rng = __import__("numpy").random.default_rng(0)
        scores = rng.random((20, 50))
        labels = (rng.random((20, 50)) > 0.7).astype("float32")
        r10 = recall_at_k(scores, labels, k=10)
        n10 = ndcg_at_k(scores, labels, k=10)
        print(f"recall@10={r10:.4f} ndcg@10={n10:.4f}")
        return 0
    if args.sub == "robustness":
        from morel.eval.protocol import robustness_sweep

        ratios = [0.1, 0.3, 0.5, 0.7]
        rng = __import__("numpy").random.default_rng(0)
        scores_by_ratio = {r: rng.random((20, 50)) for r in ratios}
        labels = (rng.random((20, 50)) > 0.7).astype("float32")
        result = robustness_sweep(
            scores_by_ratio,
            labels,
            metrics={
                "recall@10": lambda s, lst: __import__(
                    "morel.eval", fromlist=["recall_at_k"]
                ).recall_at_k(s, lst, k=10),
            },
        )
        print(result)
        return 0
    return 2


def _run_bench(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="morel bench", description="benchmark")
    parser.add_argument("--sizes", default="16,32")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args(argv)
    from morel.app import Benchmark
    from morel.core.config import Config

    sizes = [int(s) for s in args.sizes.split(",") if s]
    bench = Benchmark(
        config=Config(),
        run_dir=Path("runs") / "bench",
        sizes=sizes,
        epochs=args.epochs,
    )
    result = bench.run()
    print(result)
    return 0


def _run_reproduce(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="morel reproduce", description="reproduce")
    parser.add_argument("config", help="path to config.yaml")
    parser.add_argument("--items", type=int, default=50)
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args(argv)
    from morel.app import Reproduce

    rep = Reproduce(
        config_path=Path(args.config),
        run_dir=Path("runs") / "reproduce",
        items=args.items,
        users=args.users,
        epochs=args.epochs,
    )
    return int(bool(rep.run()))


def _run_render_fidelity(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="morel render-fidelity", description="render fidelity")
    parser.add_argument("markdown", help="output markdown path")
    parser.add_argument("json", nargs="?", default=None, help="optional output json path")
    args = parser.parse_args(argv)
    from morel.core.fidelity import render_json, render_markdown

    render_markdown(Path(args.markdown))
    if args.json is not None:
        render_json(Path(args.json))
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
    except ImportError:
        print("morel serve requires uvicorn; pip install morel[serve]", file=sys.stderr)
        return 1
    from morel.serve.app import create

    app = create()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _resolve_config_path(argv: list[str]) -> Path | None:
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            return Path(argv[i + 1])
    return None


def _load_config_or_default(path: Path | None):
    from morel.core.config import Config

    if path is None:
        return Config()
    return Config.from_yaml(path)


if __name__ == "__main__":
    raise SystemExit(main())
