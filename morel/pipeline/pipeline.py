"""End-to-end Pipeline: retrieve -> encode -> route -> codebook -> complete -> recommend.

Composes the five stages of the GRE-MC method into a single nn.Module so that
training and inference have one canonical entry point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from morel.codebook import GumbelVQ
from morel.complete import Decoders
from morel.core.config import Config
from morel.encode import Transformer
from morel.graph import Laplace
from morel.recommend import Light
from morel.retrieve import Result, retrieve, retrieve_batch
from morel.route import Top


@dataclass
class Output:
    """Pipeline forward output."""

    completed: dict[str, torch.Tensor]
    routing: torch.Tensor
    subgraph_indices: np.ndarray | None = None
    subgraph_mask: np.ndarray | None = None


class Pipeline(nn.Module):
    """End-to-end GRE-MC pipeline."""

    def __init__(
        self,
        config: Config,
        *,
        dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        if dims is None:
            dims = {"visual": 16, "text": 8}
        self.dims = dims
        self.pe_encoder = Laplace(k=config.encode.pe)
        self.transformer = Transformer(
            dims=dims,
            pe_dim=config.encode.pe,
            hidden=config.encode.hidden,
            layers=config.encode.layers,
            heads=config.encode.heads,
            dropout=config.encode.dropout,
        )
        self.router = Top(
            dim=config.encode.hidden,
            k=config.codebook.size,
            p=min(config.route.p, config.codebook.size),
            tau=config.route.tau,
        )
        self.codebook = GumbelVQ(
            dim=config.encode.hidden,
            size=config.codebook.size,
            router=self.router,
        )
        self.decoders = Decoders(
            latent_dim=config.encode.hidden,
            dims=dims,
            hidden=config.complete.hidden,
        )
        self.recommender: nn.Module | None = None
        self._retrieval_features: dict[str, np.ndarray] | None = None
        self._retrieval_mask: np.ndarray | None = None
        self._retrieval_adj: sp.csr_matrix | None = None

    def attach_recommender(self, ui_graph: sp.csr_matrix) -> None:
        """Attach a downstream LightGCN recommender sized for the graph."""
        users = ui_graph.shape[0]
        items = ui_graph.shape[1]
        self.recommender = Light(
            users=users,
            items=items,
            embed=self.config.recommend.embed,
            layers=self.config.recommend.layers,
        )
        self.recommender.adjacency(ui_graph)

    def attach_corpus(
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
        self._retrieval_features = features
        self._retrieval_mask = mask
        self._retrieval_adj = adjacency

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
            training: Whether to apply Gumbel noise.

        Returns:
            Output with completed modalities, routing weights, and (when
            retrieval buffers are set) the subgraph indices used.
        """
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
            self._retrieval_features is not None
            and self._retrieval_mask is not None
            and index is not None
        ):
            queries = [int(i) for i in index.detach().cpu().tolist()]
            result = retrieve_batch(
                queries,
                self._retrieval_features,
                self._retrieval_mask,
                self._retrieval_adj if self._retrieval_adj is not None else sp.csr_matrix(adjacency),
                anchors=self.config.retrieve.anchors,
                iters=self.config.retrieve.iters,
            )
            subgraph_indices = result.nodes
            subgraph_mask = result.mask
            hidden = self._encode_subgraph(
                features,
                mask,
                pe_full,
                result,
                device=device,
            )
        else:
            batch_size = mask.shape[0]
            if index is not None:
                pe_batch = pe_full[index]
            else:
                pe_batch = pe_full[:batch_size]
            hidden = self.transformer(features, mask, pe_batch)

        quantized, routing = self.codebook(hidden, training=training)
        completed = self.decoders(quantized, mask=mask)
        return Output(
            completed=completed,
            routing=routing,
            subgraph_indices=subgraph_indices,
            subgraph_mask=subgraph_mask,
        )

    def _encode_subgraph(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
        pe_full: torch.Tensor,
        result: Result,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode each query item with its retrieved subgraph as one sequence."""
        embeddings: list[torch.Tensor] = []
        for b in range(result.batch):
            size = int(result.sizes[b])
            if size == 0:
                node_ids = torch.zeros(1, dtype=torch.long, device=device)
                attention = torch.ones(1, dtype=torch.bool, device=device)
                node_features = {name: torch.zeros(1, d, device=device) for name, d in self.dims.items()}
                node_mask = torch.ones(1, len(self.dims), device=device)
                pe = torch.zeros(1, pe_full.shape[-1], device=device)
            else:
                node_ids_np = result.nodes[b, :size]
                node_ids = torch.from_numpy(node_ids_np).long().to(device)
                attention = torch.from_numpy(result.mask[b, :size]).bool().to(device)
                node_features = {
                    name: torch.from_numpy(self._retrieval_features[name][node_ids_np]).to(device).float()
                    for name in self.dims
                }
                node_mask = torch.from_numpy(self._retrieval_mask[node_ids_np]).to(device).float()
                pe = pe_full[node_ids]
            token = self.transformer(node_features, node_mask, pe, attention_mask=attention, sequence=True)
            embeddings.append(token)
        return torch.cat(embeddings, dim=0)


__all__ = ["Pipeline", "Output"]
