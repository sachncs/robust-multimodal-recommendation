"""Fidelity registry: every paper component declares its fidelity here.

Each entry maps a paper claim to its implementation location, the test
that proves the behavior, and any documented deviation. The
:func:`register_all` function is idempotent and is called from
``morel.core.__init__`` during package import.
"""

from __future__ import annotations

from morel.core.fidelity import Entry, register

REGISTRATION_DONE = False


def register_all() -> None:
    """Register every paper-component fidelity entry. Idempotent."""
    global REGISTRATION_DONE
    if REGISTRATION_DONE:
        return
    REGISTRATION_DONE = True

    register(
        Entry(
            name="ACS",
            paper="GRE-MC Algorithm 1",
            equation="multi-source BFS with reachability bitmask",
            status="EXACT",
            implementation="morel.retrieve.acs.compute",
            test="tests/research/test_paper.py::test_acs_bitmask_correctness_on_path",
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
            test="tests/unit/retrieve/test_anchor.py",
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
            test="tests/unit/retrieve/test_mage.py",
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
            test="tests/unit/data/test_build.py",
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
            test="tests/unit/data/test_build.py::test_kcore_shrinks_until_min_degree",
            deviation=None,
        )
    )

    register(
        Entry(
            name="Item graph construction",
            paper="GRE-MC Section 4",
            equation="sign(U^T U) with no self-loops",
            status="EXACT",
            implementation="morel.data.build.item_cooccurrence",
            test="tests/unit/data/test_build.py::test_item_cooccurrence_no_self_loops",
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
            test="tests/unit/data/test_mask.py",
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
            test="tests/unit/graph/test_laplacian.py",
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
            test="tests/unit/encode/test_encode.py::test_layer_is_preln",
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
            test="tests/unit/route/test_router.py",
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
            test="tests/unit/route/test_router.py",
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
            implementation="morel.codebook.codebook.GumbelVQ",
            test="tests/unit/codebook/test_codebook.py",
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
            test="tests/unit/codebook/test_codebook.py::test_usage_loss_zero_at_uniform",
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
            test="tests/unit/codebook/test_codebook.py::test_balance_loss_at_uniform",
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
            test="tests/unit/train/test_loss.py",
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
            test="tests/unit/recommend/test_recommend.py::test_light_l0_equals_dot",
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
            test="tests/unit/recommend/test_recommend.py::test_bpr_loss_positive",
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
            test="tests/unit/recommend/test_recommend.py::test_negatives_strict",
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
            test="tests/unit/eval/test_eval.py",
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
            test="tests/unit/eval/test_eval.py",
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
            test="tests/unit/complete/test_decoders.py",
            deviation=None,
        )
    )

    register(
        Entry(
            name="Online k-core approximation",
            paper="streaming adaptation of GRE-MC Section 4",
            equation="rolling-window online degree filter",
            status="APPROXIMATE",
            implementation="morel.data.stream.streaming_interactions",
            test="tests/unit/data/test_stream.py",
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
            implementation="morel.serve.update.PipelineUpdater",
            test="tests/unit/serve/test_serve_features.py",
            deviation="Not a closed-form online-learning algorithm. Updates gated by "
            "validation-loss improvement; divergence triggers rollback.",
        )
    )


__all__ = ["register_all"]
