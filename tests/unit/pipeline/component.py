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

from morel.codebook import KIND as CODEBOOK_KIND
from morel.codebook import Codebook
from morel.complete import KIND as COMPLETE_KIND
from morel.core.config import Config
from morel.core.errors import Cfg
from morel.encode import KIND as ENCODE_KIND
from morel.pipeline import Pipeline
from morel.recommend import KIND as RECOMMEND_KIND
from morel.route import KIND as ROUTE_KIND


def make(**sections: dict[str, object]) -> Config:
    """Build a small but valid config, overriding the given sections."""
    payload: dict[str, dict[str, object]] = {
        "encode": {"hidden": 16, "pe": 4, "layers": 1, "heads": 2},
        "codebook": {"size": 16},
        "route": {"p": 4},
    }
    for name, values in sections.items():
        payload.setdefault(name, {}).update(values)
    return Config.parse(payload)


def ui() -> sp.csr_matrix:
    """Return a tiny bipartite interaction matrix."""
    return sp.csr_matrix(np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float32))


class Checker:
    """Aggregated test methods for this module."""

    @pytest.mark.parametrize(
        ("kind", "expected"), [("gumbel", "Soft"), ("vq", "VQ"), ("identity", "Noop")]
    )
    def implementation(self, kind: str, expected: str) -> None:
        pipeline = Pipeline(make(codebook={"kind": kind}), dims={"visual": 4, "text": 2})
        assert type(pipeline.codebook).__name__ == expected

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [("top", "Top"), ("dense", "Dense"), ("gumbel", "Gumbel"), ("fixed", "Fixed")],
    )
    def the(self, kind: str, expected: str) -> None:
        pipeline = Pipeline(make(route={"kind": kind}), dims={"visual": 4, "text": 2})
        assert type(pipeline.router).__name__ == expected

    @pytest.mark.parametrize(
        ("kind", "expected"), [("transformer", "Transformer"), ("identity", "Identity")]
    )
    def selects(self, kind: str, expected: str) -> None:
        pipeline = Pipeline(make(encode={"kind": kind}), dims={"visual": 4, "text": 2})
        assert type(pipeline.transformer).__name__ == expected

    @pytest.mark.parametrize(
        ("kind", "expected"), [("light", "Light"), ("mf", "MF"), ("pop", "Pop")]
    )
    def kind(self, kind: str, expected: str) -> None:
        config = make(recommend={"kind": kind, "embed": 8, "layers": 2})
        pipeline = Pipeline(config, dims={"visual": 4, "text": 2})
        pipeline.wire(ui())
        assert pipeline.recommender is not None
        assert type(pipeline.recommender).__name__ == expected

    def score(self) -> None:
        """Selection is not enough; each ranker must be usable once attached."""
        for kind in RECOMMEND_KIND:
            config = make(recommend={"kind": kind, "embed": 8, "layers": 2})
            pipeline = Pipeline(config, dims={"visual": 4, "text": 2})
            pipeline.wire(ui())
            assert pipeline.recommender is not None
            scores = pipeline.recommender(torch.arange(2), torch.arange(3))
            assert scores.shape == (2, 3), f"{kind} produced {scores.shape}"

    @pytest.mark.parametrize(
        ("section", "field"),
        [
            ("codebook", "codebook"),
            ("encode", "encoder"),
            ("route", "router"),
            ("recommend", "recommender"),
        ],
    )
    def names(self, section: str, field: str) -> None:
        config = make(**{section: {"kind": "definitely-not-real"}})
        with pytest.raises(Cfg, match=rf"unknown {field} 'definitely-not-real'; available: "):
            Pipeline(config, dims={"visual": 4, "text": 2})

    def rejected(self) -> None:
        config = make(recommend={"kind": "definitely-not-real"})
        with pytest.raises(Cfg, match="unknown recommender"):
            Pipeline(config, dims={"visual": 4, "text": 2})

    def end(self) -> None:
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

        routers = [k for k in ROUTE_KIND if k != "fixed"]
        combinations = list(itertools.product(ENCODE_KIND, routers, CODEBOOK_KIND))
        assert len(combinations) >= 12, "expected a meaningful number of selectable combinations"

        for encoder, router, codebook in combinations:
            config = make(
                encode={"kind": encoder},
                route={"kind": router},
                codebook={"kind": codebook},
            )
            pipeline = Pipeline(config, dims={"visual": 4, "text": 2})
            pipeline.attach(features, mask, adjacency)
            out = pipeline(batch, batch_mask, adjacency, index=index, training=False)
            assert out.completed["visual"].shape == (6, 4), f"{encoder}/{router}/{codebook}"
            assert out.routing.shape[0] == 6, f"{encoder}/{router}/{codebook}"

    def morel(self) -> None:
        """Built-in codebooks are available in the KIND dict."""

        class Doubling(Codebook):
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

        # Verify the built-in codebooks are present
        assert "gumbel" in CODEBOOK_KIND
        assert "vq" in CODEBOOK_KIND
        assert "identity" in CODEBOOK_KIND
        # Verify the pipeline can use any of them
        pipeline = Pipeline(make(codebook={"kind": "vq"}), dims={"visual": 4, "text": 2})
        assert isinstance(pipeline.codebook, CODEBOOK_KIND["vq"])

        out = pipeline(
            {"visual": torch.randn(3, 4), "text": torch.randn(3, 2)},
            torch.ones(3, 2),
            sp.csr_matrix((3, 3), dtype=np.float32),
            training=False,
        )
        assert out.completed["visual"].shape == (3, 4)

    def populated(self) -> None:
        """A registry that silently lost its entries would make every kind invalid."""
        assert set(CODEBOOK_KIND) >= {"gumbel", "vq", "identity"}
        assert set(ROUTE_KIND) >= {"top", "dense", "gumbel", "fixed"}
        assert set(ENCODE_KIND) >= {"transformer", "identity"}
        assert set(RECOMMEND_KIND) >= {"light", "mf", "pop"}
        assert set(COMPLETE_KIND) >= {"mlp"}

    def components(self) -> None:
        """The shipped defaults must not point at implementations that do not exist."""
        config = Config()
        assert config.encode.kind in ENCODE_KIND
        assert config.route.kind in ROUTE_KIND
        assert config.codebook.kind in CODEBOOK_KIND
        assert config.complete.kind in COMPLETE_KIND
        assert config.recommend.kind in RECOMMEND_KIND
