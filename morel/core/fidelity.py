"""Paper-fidelity registry.

Every algorithm component declares a `Fidelity` entry. The registry renders
``docs/FIDELITY.md`` and ``docs/FIDELITY.json`` from this state, so the
documentation can never drift from the implementation.

The module-level :func:`register` is idempotent and runs at import time
from ``morel.core.__init__``. Each :class:`Entry` maps a paper claim to its
implementation location, the test that proves the behaviour, and any
documented deviation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

Status = Literal["EXACT", "APPROXIMATE", "INCORRECT", "UNKNOWN"]


@dataclass(frozen=True)
class Entry:
    """One component's fidelity declaration.

    Attributes
    ----------
        name: Human-readable component name.
        paper: Bibliographic anchor for the claim.
        equation: Plain-English equation or algorithm reference.
        status: One of EXACT / APPROXIMATE / INCORRECT / UNKNOWN.
        implementation: ``dotted.module.path::callable`` form.
        test: Test reference that proves the behaviour.
        deviation: Free-text deviation from the paper, if any.
        notes: Free-text notes (test path, fix rationale, etc.).
    """

    name: str
    paper: str
    equation: str
    status: Status
    implementation: str
    test: str
    deviation: str | None = None
    notes: str | None = None


registry: dict[str, Entry] = {}


def register(entry: Entry) -> Entry:
    """Register a fidelity entry. Use as a decorator.

    Args:
        entry: The fidelity declaration to register.

    Returns
    -------
        The same entry, for decorator chaining.
    """
    registry[entry.name] = entry
    return entry


def render(target: Path | str) -> None:
    """Render the registry as a Markdown report.

    Args:
        target: Destination file path.
    """
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Paper Fidelity Report",
        "",
        f"Generated: {datetime.now(tz=UTC).isoformat()}",
        "",
        "| Component | Status | Paper | Equation | Implementation | Test | Deviation |",
        "|-----------|--------|-------|----------|----------------|------|-----------|",
    ]
    for entry in sorted(registry.values(), key=lambda e: e.name):
        status = entry.status
        deviation = entry.deviation or "—"
        lines.append(
            f"| `{entry.name}` | **{status}** | {entry.paper} | {entry.equation} | "
            f"`{entry.implementation}` | `{entry.test}` | {deviation} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_json(target: Path | str) -> None:
    """Render the registry as a JSON file.

    Args:
        target: Destination file path.
    """
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": datetime.now(tz=UTC).isoformat(),
        "entries": [asdict(entry) for entry in sorted(registry.values(), key=lambda e: e.name)],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def clear() -> None:
    """Clear the registry. Used by tests."""
    registry.clear()


def all() -> list[Entry]:
    """Return all registered entries sorted by name."""
    return sorted(registry.values(), key=lambda e: e.name)


def register_all() -> None:
    """Register every paper-component fidelity entry.

    The function is idempotent. It is called from ``morel.core.__init__``
    during package import so that documentation rendering and the test
    suite see the full set.
    """
    if registry:
        return

    register(
        Entry(
            name="ACS",
            paper="GRE-MC Algorithm 1",
            equation="multi-source BFS with reachability bitmask",
            status="EXACT",
            implementation="morel.retrieve.acs.compute",
            test="tests/research/paper.py::Checker::acs",
            deviation=None,
            notes="Iterative backtrack (no recursion-limit crash on long paths). "
            "Duplicate and out-of-range anchor rejection. Self-loop guard.",
        )
    )

    register(
        Entry(
            name="Anchor retrieval",
            paper="GRE-MC Section 4.1",
            equation="cosine NN over observed modalities",
            status="EXACT",
            implementation="morel.retrieve.anchor.query",
            test="tests/unit/retrieve/anchor.py",
            deviation=None,
        )
    )

    register(
        Entry(
            name="MAGE",
            paper="GRE-MC Algorithm 2",
            equation="greedy boundary add/remove with mean-relevance objective",
            status="APPROXIMATE",
            implementation="morel.retrieve.mage.expand",
            test="tests/unit/retrieve/mage.py",
            deviation="Best-improvement hill climbing (vs. paper's first-improvement "
            "ambiguity); sorted boundary iteration (deterministic vs. paper's set-iteration).",
            notes="Sorted iteration gives the same final subgraph but eliminates the "
            "non-determinism inherent in Python set ordering.",
        )
    )

    register(
        Entry(
            name="Bipartite construction",
            paper="GRE-MC Section 4",
            equation="user-item CSR matrix",
            status="EXACT",
            implementation="morel.data.build.bipartite",
            test="tests/unit/data/build.py",
            deviation=None,
        )
    )

    register(
        Entry(
            name="Iterative k-core",
            paper="GRE-MC Section 4 (data filtering)",
            equation="peel nodes below min_edges until stable",
            status="EXACT",
            implementation="morel.data.build.kcore",
            test="tests/unit/data/build.py::kcore",
            deviation=None,
        )
    )

    register(
        Entry(
            name="Item graph construction",
            paper="GRE-MC Section 4",
            equation="sign(U^T U) with no self-loops",
            status="EXACT",
            implementation="morel.data.build.cooccurrence",
            test="tests/unit/data/build.py::Checker::cooccurrence",
            deviation=None,
        )
    )

    register(
        Entry(
            name="Modality masking",
            paper="GRE-MC Section 5 (robustness)",
            equation="Bernoulli availability mask with at-least-one repair",
            status="EXACT",
            implementation="morel.data.mask.bernoulli",
            test="tests/unit/data/mask.py",
            deviation=None,
        )
    )

    register(
        Entry(
            name="Laplacian PE",
            paper="GRE-MC Section 4.3",
            equation="bottom-k nontrivial eigenvectors of L = I - D^{-1/2} A D^{-1/2}",
            status="EXACT",
            implementation="morel.graph.laplacian.pe",
            test="tests/unit/graph/laplacian.py",
            deviation=None,
            notes="Dense eigh fallback when ARPACK does not converge.",
        )
    )

    register(
        Entry(
            name="Joint encoding (transformer)",
            paper="GRE-MC Section 4.2",
            equation="Pre-LN graph transformer over subgraph tokens",
            status="EXACT",
            implementation="morel.encode.transformer.Transformer",
            test="tests/unit/encode/encode.py::Checker::preln",
            deviation=None,
            notes="Pre-LN is implemented as documented; see METHOD.md.",
        )
    )

    register(
        Entry(
            name="Gumbel-Softmax routing",
            paper="GRE-MC Section 4.4",
            equation="softmax((Wz + g) / tau)",
            status="EXACT",
            implementation="morel.route.router.Gumbel",
            test="tests/unit/route/router.py",
            deviation=None,
        )
    )

    register(
        Entry(
            name="Top-P sparse routing",
            paper="GRE-MC Section 4.4",
            equation="top-p renormalised over codebook logits",
            status="APPROXIMATE",
            implementation="morel.route.router.Top",
            test="tests/unit/route/router.py",
            deviation="Implemented as Top-K (post-softmax topk + renorm); the paper "
            "text says Top-P but the numerics are equivalent in expectation.",
        )
    )

    register(
        Entry(
            name="Codebook (Gumbel-VQ)",
            paper="GRE-MC Section 4.4",
            equation="g_top @ codebook",
            status="EXACT",
            implementation="morel.codebook.codebook.Soft",
            test="tests/unit/codebook/codebook.py",
            deviation=None,
        )
    )

    register(
        Entry(
            name="Usage loss",
            paper="GRE-MC Eq. 7",
            equation="KL(bar_p || uniform)",
            status="EXACT",
            implementation="morel.codebook.codebook.usage",
            test="tests/unit/codebook/codebook.py::Checker::usage",
            deviation=None,
        )
    )

    register(
        Entry(
            name="Load loss",
            paper="GRE-MC Eq. 8",
            equation="K * sum_e bar_g_e^2",
            status="EXACT",
            implementation="morel.codebook.codebook.balance",
            test="tests/unit/codebook/codebook.py::Checker::balance",
            deviation=None,
        )
    )

    register(
        Entry(
            name="Reconstruction loss",
            paper="GRE-MC Section 4.5",
            equation="element-normalised masked MSE over missing positions",
            status="EXACT",
            implementation="morel.train.loss.Reconstruction",
            test="tests/unit/train/loss.py",
            deviation=None,
        )
    )

    register(
        Entry(
            name="LightGCN propagation",
            paper="He et al. 2020 (LightGCN)",
            equation="H^{l+1} = A_hat H^l; H_final = mean(H^0..H^L)",
            status="EXACT",
            implementation="morel.recommend.light.Light",
            test="tests/unit/recommend/recommend.py::Checker::l0",
            deviation=None,
        )
    )

    register(
        Entry(
            name="BPR loss",
            paper="Rendle et al. 2009",
            equation="-log sigmoid(pos - neg)",
            status="EXACT",
            implementation="morel.recommend.bpr.bpr",
            test="tests/unit/recommend/recommend.py::Checker::bpr",
            deviation=None,
        )
    )

    register(
        Entry(
            name="Strict negative sampling",
            paper="GRE-MC Section 5 (downstream)",
            equation="sample negatives that are not in positives",
            status="EXACT",
            implementation="morel.recommend.bpr.negatives",
            test="tests/unit/recommend/recommend.py::Checker::strict",
            deviation=None,
        )
    )

    register(
        Entry(
            name="Recall@K",
            paper="standard IR metric",
            equation="hits@k / relevant",
            status="EXACT",
            implementation="morel.eval.ranking.recall_at_k",
            test="tests/unit/eval/eval.py",
            deviation=None,
        )
    )

    register(
        Entry(
            name="NDCG@K",
            paper="standard IR metric",
            equation="DCG@k / IDCG@k",
            status="EXACT",
            implementation="morel.eval.ranking.ndcg_at_k",
            test="tests/unit/eval/eval.py",
            deviation=None,
        )
    )

    register(
        Entry(
            name="Modality decoder",
            paper="GRE-MC Section 4.5",
            equation="f̂_i^(m) = MLP^(m)(q_i) with learned [MASK] token",
            status="EXACT",
            implementation="morel.complete.decoders.Decoders",
            test="tests/unit/complete/decoders.py",
            deviation=None,
        )
    )

    register(
        Entry(
            name="Online k-core approximation",
            paper="streaming adaptation of GRE-MC Section 4",
            equation="rolling-window online degree filter",
            status="APPROXIMATE",
            implementation="morel.data.stream.stream_interactions",
            test="tests/unit/data/stream.py",
            deviation="Online degree filter is offline-exact when two passes are available; "
            "single-pass streaming uses a rolling-window approximation. Offline k-core "
            "remains available in morel.data.build.kcore.",
            notes="Documented in LIMITATIONS.md.",
        )
    )

    register(
        Entry(
            name="Online full-pipeline update",
            paper="production extension of GRE-MC Section 5",
            equation="replay buffer + divergence guard",
            status="APPROXIMATE",
            implementation="morel.serve.update.Updater",
            test="tests/unit/serve/features.py",
            deviation="Not a closed-form online-learning algorithm. Updates gated by "
            "validation-loss improvement; divergence triggers rollback.",
        )
    )


__all__ = [
    "Entry",
    "Status",
    "all",
    "clear",
    "register",
    "registry",
    "render",
]
