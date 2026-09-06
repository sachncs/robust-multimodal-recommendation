"""The configuration must select pipeline components, and be extensible.

Every stage of the pipeline carries a ``kind`` in the configuration. Those
fields used to be dead: ``Pipeline`` hardcoded one implementation per stage and
ignored them, so a config asking for the no-codebook ablation silently got the
full model. These tests pin that the config now chooses, that an unknown choice
fails loudly, and that a component registered from outside morel is usable
without editing the package.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from morel.codebook import CODEBOOKS, Codebook
from morel.complete import COMPLETERS
from morel.core.config import Config
from morel.core.errors import ConfigError
from morel.encode import ENCODERS
from morel.pipeline import Pipeline
from morel.recommend import RECOMMENDERS
from morel.route import ROUTERS


def make_config(**sections: dict[str, object]) -> Config:
    """Build a small but valid config, overriding the given sections."""
    payload: dict[str, dict[str, object]] = {
        "encode": {"hidden": 16, "pe": 4, "layers": 1, "heads": 2},
        "codebook": {"size": 16},
        "route": {"p": 4},
    }
    for name, values in sections.items():
        payload.setdefault(name, {}).update(values)
    return Config.from_dict(payload)


def ui_graph() -> sp.csr_matrix:
    """Return a tiny bipartite interaction matrix."""
    return sp.csr_matrix(np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float32))


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("gumbel", "GumbelVQ"), ("vq", "VQ"), ("identity", "IdentityCodebook")],
)
def test_codebook_kind_selects_the_implementation(kind: str, expected: str) -> None:
    pipeline = Pipeline(make_config(codebook={"kind": kind}), dims={"visual": 4, "text": 2})
    assert type(pipeline.codebook).__name__ == expected


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("top", "Top"), ("dense", "Dense"), ("gumbel", "Gumbel"), ("fixed", "Fixed")],
)
def test_route_kind_selects_the_implementation(kind: str, expected: str) -> None:
    pipeline = Pipeline(make_config(route={"kind": kind}), dims={"visual": 4, "text": 2})
    assert type(pipeline.router).__name__ == expected


@pytest.mark.parametrize(
    ("kind", "expected"), [("transformer", "Transformer"), ("identity", "Identity")]
)
def test_encode_kind_selects_the_implementation(kind: str, expected: str) -> None:
    pipeline = Pipeline(make_config(encode={"kind": kind}), dims={"visual": 4, "text": 2})
    assert type(pipeline.transformer).__name__ == expected


@pytest.mark.parametrize(("kind", "expected"), [("light", "Light"), ("mf", "MF"), ("pop", "Pop")])
def test_recommend_kind_selects_the_implementation(kind: str, expected: str) -> None:
    config = make_config(recommend={"kind": kind, "embed": 8, "layers": 2})
    pipeline = Pipeline(config, dims={"visual": 4, "text": 2})
    pipeline.attach_recommender(ui_graph())
    assert pipeline.recommender is not None
    assert type(pipeline.recommender).__name__ == expected


def test_every_attached_recommender_can_score() -> None:
    """Selection is not enough; each ranker must be usable once attached."""
    for kind in RECOMMENDERS.available():
        config = make_config(recommend={"kind": kind, "embed": 8, "layers": 2})
        pipeline = Pipeline(config, dims={"visual": 4, "text": 2})
        pipeline.attach_recommender(ui_graph())
        assert pipeline.recommender is not None
        scores = pipeline.recommender(torch.arange(2), torch.arange(3))
        assert scores.shape == (2, 3), f"{kind} produced {scores.shape}"


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("codebook", "codebook"),
        ("route", "router"),
        ("encode", "encoder"),
        ("complete", "completer"),
    ],
)
def test_unknown_kind_is_rejected_with_the_available_names(section: str, field: str) -> None:
    config = make_config(**{section: {"kind": "definitely-not-real"}})
    with pytest.raises(ConfigError, match=rf"unknown {field} 'definitely-not-real'; available: "):
        Pipeline(config, dims={"visual": 4, "text": 2})


def test_unknown_recommender_kind_is_rejected() -> None:
    config = make_config(recommend={"kind": "definitely-not-real"})
    pipeline = Pipeline(config, dims={"visual": 4, "text": 2})
    with pytest.raises(ConfigError, match="unknown recommender"):
        pipeline.attach_recommender(ui_graph())


def test_registered_combinations_all_run_end_to_end() -> None:
    """Any selectable encoder/router/codebook triple must survive a forward pass."""
    nodes = 32
    rng = np.random.default_rng(0)
    features = {
        "visual": rng.normal(size=(nodes, 4)).astype(np.float32),
        "text": rng.normal(size=(nodes, 2)).astype(np.float32),
    }
    mask = np.ones((nodes, 2), dtype=np.float32)
    arr = np.zeros((nodes, nodes), dtype=np.float32)
    for i in range(nodes - 1):
        arr[i, i + 1] = arr[i + 1, i] = 1.0
    adjacency = sp.csr_matrix(arr)
    index = torch.arange(6)
    batch = {name: torch.from_numpy(value[:6]) for name, value in features.items()}
    batch_mask = torch.from_numpy(mask[:6])

    routers = [k for k in ROUTERS.available() if k != "fixed"]
    combinations = list(itertools.product(ENCODERS.available(), routers, CODEBOOKS.available()))
    assert len(combinations) >= 12, "expected a meaningful number of selectable combinations"

    for encoder, router, codebook in combinations:
        config = make_config(
            encode={"kind": encoder},
            route={"kind": router},
            codebook={"kind": codebook},
        )
        pipeline = Pipeline(config, dims={"visual": 4, "text": 2})
        pipeline.attach(features, mask, adjacency)
        out = pipeline(batch, batch_mask, adjacency, index=index, training=False)
        assert out.completed["visual"].shape == (6, 4), f"{encoder}/{router}/{codebook}"
        assert out.routing.shape[0] == 6, f"{encoder}/{router}/{codebook}"


def test_a_third_party_component_needs_no_change_to_morel() -> None:
    """The extension point: register a new codebook and select it from config."""

    class DoublingCodebook(Codebook):
        """A codebook defined entirely outside the morel package."""

        def __init__(self, dim: int, size: int) -> None:
            super().__init__()
            self.dim = dim
            self.size = size

        def forward(
            self, hidden: torch.Tensor, *, training: bool = True
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del training
            probs = torch.full(
                (*hidden.shape[:-1], self.size),
                1.0 / self.size,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            return hidden * 2, probs

    name = "doubling-test-only"
    CODEBOOKS.register(name, lambda *, dim, size, router, seed=None: DoublingCodebook(dim, size))
    try:
        assert name in CODEBOOKS
        pipeline = Pipeline(make_config(codebook={"kind": name}), dims={"visual": 4, "text": 2})
        assert isinstance(pipeline.codebook, DoublingCodebook)

        out = pipeline(
            {"visual": torch.randn(3, 4), "text": torch.randn(3, 2)},
            torch.ones(3, 2),
            sp.csr_matrix((3, 3), dtype=np.float32),
            training=False,
        )
        assert out.completed["visual"].shape == (3, 4)
    finally:
        CODEBOOKS.unregister(name)

    assert name not in CODEBOOKS


def test_registries_are_populated() -> None:
    """A registry that silently lost its entries would make every kind invalid."""
    assert set(CODEBOOKS.available()) >= {"gumbel", "vq", "identity"}
    assert set(ROUTERS.available()) >= {"top", "dense", "gumbel", "fixed"}
    assert set(ENCODERS.available()) >= {"transformer", "identity"}
    assert set(RECOMMENDERS.available()) >= {"light", "mf", "pop"}
    assert set(COMPLETERS.available()) >= {"mlp"}


def test_config_defaults_name_registered_components() -> None:
    """The shipped defaults must not point at implementations that do not exist."""
    config = Config()
    assert config.encode.kind in ENCODERS
    assert config.route.kind in ROUTERS
    assert config.codebook.kind in CODEBOOKS
    assert config.complete.kind in COMPLETERS
    assert config.recommend.kind in RECOMMENDERS
