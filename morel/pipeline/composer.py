"""End-to-end Pipeline: retrieve -> encode -> route -> codebook -> complete -> recommend.

Composes the five stages of the GRE-MC method into a single nn.Module so that
training and inference have one canonical entry point.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from morel.codebook import CODEBOOKS
from morel.complete import COMPLETERS
from morel.core.config import Config
from morel.core.errors import GraphError, ModelError
from morel.core.seed import deterministic
from morel.encode import ENCODERS
from morel.graph import Laplace
from morel.recommend import RECOMMENDERS
from morel.retrieve import Result, retrieve_batch
from morel.route import build as build_route


@dataclass
class Output:
    """Pipeline forward output."""

    completed: dict[str, torch.Tensor]
    routing: torch.Tensor
    subgraph_indices: np.ndarray | None = None
    subgraph_mask: np.ndarray | None = None


@contextmanager
def module_mode(module: nn.Module, training: bool) -> Iterator[None]:
    """Run a block with ``module`` in the given train/eval mode, then restore.

    ``nn.Dropout`` and friends key off :attr:`torch.nn.Module.training`, which
    a ``training=`` keyword argument on a ``forward`` method does not change.
    Without this, asking a pipeline for an inference pass still applies
    dropout, which makes the result both wrong and nondeterministic.

    Args:
        module: Module whose mode (and that of its children) is switched.
        training: ``True`` for train mode, ``False`` for eval mode.

    Yields
    ------
        ``None``. The block body runs with the requested mode applied.
    """
    previous = module.training
    module.train(training)
    try:
        yield
    finally:
        module.train(previous)


class Pipeline(nn.Module):
    """End-to-end GRE-MC pipeline."""

    def __init__(
        self,
        config: Config,
        *,
        dims: dict[str, int] | None = None,
    ) -> None:
        """Compose the GRE-MC stages into one module.

        Every parameter-bearing stage is initialized under ``config.seed``, so
        two pipelines built from the same config hold identical weights
        regardless of what the calling process did to the global RNG
        beforehand. The caller's RNG state is restored on exit.

        Each stage is selected by its ``kind`` in the configuration and built
        through the corresponding registry, so an implementation registered
        from outside this package is usable without touching this file.

        Args:
            config: Validated configuration supplying every stage's
                hyperparameters, implementation choice, and the
                initialization seed.
            dims: Per-modality input widths; defaults to the demo dims.

        Raises
        ------
            ConfigError: If any stage names an implementation that is not
                registered. The message lists the available names.
        """
        super().__init__()
        self.config = config
        if dims is None:
            dims = {"visual": 16, "text": 8}
        self.dims = dims
        self.pe_encoder = Laplace(k=config.encode.pe)
        with deterministic(config.seed):
            self.transformer = ENCODERS.create(
                config.encode.kind,
                dims=dims,
                pe_dim=config.encode.pe,
                hidden=config.encode.hidden,
                layers=config.encode.layers,
                heads=config.encode.heads,
                dropout=config.encode.dropout,
            )
            self.router = build_route(
                config.route.kind,
                dim=config.encode.hidden,
                k=config.codebook.size,
                p=min(config.route.p, config.codebook.size),
                tau=config.route.tau,
            )
            self.codebook = CODEBOOKS.create(
                config.codebook.kind,
                dim=config.encode.hidden,
                size=config.codebook.size,
                router=self.router,
            )
            self.decoders = COMPLETERS.create(
                config.complete.kind,
                latent_dim=config.encode.hidden,
                dims=dims,
                hidden=config.complete.hidden,
            )
        self.recommender: nn.Module | None = None
        self.retrieval_features: dict[str, np.ndarray] | None = None
        self.retrieval_mask: np.ndarray | None = None
        self.retrieval_adj: sp.csr_matrix | None = None

    def attach_recommender(
        self, ui_graph: sp.csr_matrix, *, feature_dim: int | None = None
    ) -> None:
        """Attach the downstream ranker named by ``config.recommend.kind``.

        The ranker is initialized under ``config.seed``. Whatever
        graph-derived state it needs is prepared here rather than on the first
        scoring call: a propagation-based ranker gets its normalized adjacency
        cached, and a popularity ranker gets fitted.

        Args:
            ui_graph: Bipartite ``(users, items)`` interaction matrix.
            feature_dim: Width of the completion output that will be handed to
                the ranker, for rankers that can consume it. Leave as ``None``
                to rank from ID embeddings and the graph alone.

        Raises
        ------
            ConfigError: If ``config.recommend.kind`` is not registered.
        """
        users = ui_graph.shape[0]
        items = ui_graph.shape[1]
        recommender = RECOMMENDERS.create(
            self.config.recommend.kind,
            users=users,
            items=items,
            embed=self.config.recommend.embed,
            layers=self.config.recommend.layers,
            feature_dim=feature_dim,
            seed=self.config.seed,
        )
        prime = getattr(recommender, "normalized_adjacency", None)
        if callable(prime):
            prime(ui_graph)
        fit = getattr(recommender, "fit", None)
        if callable(fit):
            fit(ui_graph)
        self.recommender = recommender

    def attach(
        self,
        features: dict[str, np.ndarray],
        mask: np.ndarray,
        adjacency: sp.csr_matrix | None = None,
    ) -> None:
        """Bind full-graph data for on-the-fly retrieval.

        Unlike :meth:`torch.nn.Module.register_buffers`, this does not register
        PyTorch tensors with the autograd engine. It only stores the corpus
        attributes used by retrieval during :meth:`forward`.
        """
        self.retrieval_features = features
        self.retrieval_mask = mask
        self.retrieval_adj = adjacency

    def forward(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
        adjacency: sp.spmatrix,
        index: torch.Tensor | None = None,
        training: bool = True,
    ) -> Output:
        """End-to-end forward pass.

        Args:
            features: Per-modality features ``(B, d_m)``.
            mask: ``(B, M)`` availability.
            adjacency: Sparse item-item adjacency.
            index: Optional ``(B,)`` item ids for retrieval.
            training: Whether the pass is stochastic. This governs *all*
                stochastic components, not just Gumbel noise: the encoder's
                dropout layers are switched to match for the duration of the
                call and restored afterwards. Pass ``False`` for inference to
                get a deterministic, dropout-free result.

        Returns
        -------
            Output with completed modalities, routing weights, and (when
            retrieval buffers are set) the subgraph indices used.

        Raises
        ------
            GraphError: If the adjacency has self-loops. Build the item graph
                once with ``morel.data.build.item_cooccurrence``; that
                builder removes self-loops before persisting.
        """
        if adjacency.diagonal().any():
            raise GraphError(
                "Pipeline.forward received an adjacency with self-loops; "
                "build the item graph once with "
                "morel.data.build.item_cooccurrence (which removes them) "
                "and pass that adjacency here."
            )
        with module_mode(self, training):
            device = features[next(iter(features))].device
            pe_full = self.pe_encoder(adjacency).to(device)
            # Pad PE up to configured pe_dim if the graph is too small to provide
            # the requested number of nontrivial eigenvectors.
            if pe_full.shape[-1] < self.config.encode.pe:
                pad = torch.zeros(
                    *pe_full.shape[:-1],
                    self.config.encode.pe - pe_full.shape[-1],
                    device=device,
                    dtype=pe_full.dtype,
                )
                pe_full = torch.cat([pe_full, pad], dim=-1)
            subgraph_indices: np.ndarray | None = None
            subgraph_mask: np.ndarray | None = None

            if (
                self.retrieval_features is not None
                and self.retrieval_mask is not None
                and index is not None
            ):
                queries = [int(i) for i in index.detach().cpu().tolist()]
                result = retrieve_batch(
                    queries,
                    self.retrieval_features,
                    self.retrieval_mask,
                    self.retrieval_adj
                    if self.retrieval_adj is not None
                    else sp.csr_matrix(adjacency),
                    anchors=self.config.retrieve.anchors,
                    iters=self.config.retrieve.iters,
                    kind=self.config.retrieve.kind,
                )
                subgraph_indices = result.nodes
                subgraph_mask = result.mask
                hidden = self.encode_subgraph(
                    features,
                    mask,
                    pe_full,
                    result,
                    device=device,
                )
            else:
                batch_size = mask.shape[0]
                pe_batch = pe_full[index] if index is not None else pe_full[:batch_size]
                hidden = self.transformer(features, mask, pe_batch)

            quantized, routing = self.codebook(hidden, training=training)
            completed = self.decoders(quantized, mask=mask)
        return Output(
            completed=completed,
            routing=routing,
            subgraph_indices=subgraph_indices,
            subgraph_mask=subgraph_mask,
        )

    def encode_subgraph(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
        pe_full: torch.Tensor,
        result: Result,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode each query item with its retrieved subgraph as one padded batch.

        All per-query subgraphs are padded to ``result.max_size`` and processed
        in a single transformer call with a batched attention mask, instead of
        one Python loop iteration per query. Empty subgraphs are padded with a
        single attention-masked token so the transformer still produces one
        embedding per query.

        Raises
        ------
            ModelError: If no corpus has been attached. Call
                :meth:`attach` before encoding subgraphs.
        """
        corpus_features = self.retrieval_features
        corpus_mask = self.retrieval_mask
        if corpus_features is None or corpus_mask is None:
            raise ModelError(
                "encode_subgraph needs a bound corpus; "
                "call Pipeline.attach(features, mask, adjacency) first"
            )
        max_size = max(int(result.max_size), 1)
        batch = int(result.batch)
        modalities = list(self.dims.keys())
        node_features: dict[str, torch.Tensor] = {}
        node_mask = torch.zeros(batch, max_size, len(modalities), device=device)
        pe = torch.zeros(batch, max_size, pe_full.shape[-1], device=device)
        attention = torch.zeros(batch, max_size, dtype=torch.bool, device=device)
        for name in modalities:
            node_features[name] = torch.zeros(batch, max_size, self.dims[name], device=device)

        for b in range(batch):
            size = int(result.sizes[b])
            if size == 0:
                attention[b, 0] = True
                node_mask[b, 0, :] = 1.0
                continue
            node_ids_np = result.nodes[b, :size]
            for name in modalities:
                node_features[name][b, :size] = (
                    torch.from_numpy(corpus_features[name][node_ids_np]).to(device).float()
                )
            node_mask[b, :size] = torch.from_numpy(corpus_mask[node_ids_np]).to(device).float()
            pe[b, :size] = pe_full[node_ids_np]
            attention[b, :size] = torch.from_numpy(result.mask[b, :size]).bool().to(device)

        hidden: torch.Tensor = self.transformer(
            node_features,
            node_mask,
            pe,
            attention_mask=attention,
            sequence=True,
        )
        return hidden


__all__ = ["Output", "Pipeline", "module_mode"]
