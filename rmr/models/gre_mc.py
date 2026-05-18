"""End-to-end GRE-MC model.

Composes a Laplacian positional encoding module, a joint-encoding graph
transformer, a sparse-routing codebook, and per-modality decoders to
reconstruct missing modalities.  When retrieval buffers are registered,
the model performs modality-aware subgraph retrieval (anchor retrieval +
ACS + MAGE) and jointly encodes the query node with the retrieved
subgraph.
"""


import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from rmr.models.codebook import SparseRoutingCodebook
from rmr.models.decoder import ModalityDecoders
from rmr.models.positional_encoding import LaplacianPE
from rmr.models.retrieval import anchor_retrieval, mage
from rmr.models.transformer import JointEncodingGraphTransformer


class GREMC(nn.Module):
    """End-to-end GRE-MC model: transformer -> codebook -> decoders."""

    def __init__(
        self,
        input_dims: dict[str, int],
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        codebook_size: int = 100,
        top_p: int = 4,
        tau: float = 0.5,
        pe_dim: int = 20,
        dropout: float = 0.5,
        top_k_anchors: int = 10,
        mage_iters: int = 10,
    ) -> None:
        """Initializes the GRE-MC model.

        Args:
            input_dims: Mapping from modality name to feature dimension.
            hidden_dim: Hidden dimensionality.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            codebook_size: Number of codebook entries.
            top_p: Number of top codebook entries to select.
            tau: Gumbel-Softmax temperature.
            pe_dim: Dimensionality of Laplacian positional encodings.
            dropout: Dropout probability.
            top_k_anchors: Number of anchors to retrieve per observed modality.
            mage_iters: Maximum MAGE expansion iterations.
        """
        super().__init__()
        self.pe_encoder = LaplacianPE(k=pe_dim)
        self.transformer = JointEncodingGraphTransformer(
            input_dims=input_dims,
            pe_dim=pe_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.codebook = SparseRoutingCodebook(
            input_dim=hidden_dim,
            codebook_size=codebook_size,
            top_p=top_p,
            tau=tau,
        )
        self.decoder = ModalityDecoders(
            latent_dim=hidden_dim,
            output_dims=input_dims,
        )
        self.top_k_anchors = top_k_anchors
        self.mage_iters = mage_iters
        self._retrieval_features: dict[str, np.ndarray] | None = None
        self._retrieval_mask: np.ndarray | None = None

    def register_retrieval_buffers(
        self,
        all_features: dict[str, np.ndarray],
        all_mask: np.ndarray,
    ) -> None:
        """Store full item features and masks for on-the-fly retrieval.

        Args:
            all_features: Dictionary mapping modality names to feature arrays
                of shape ``(n_items, d_m)``.
            all_mask: Binary mask array of shape ``(n_items, n_modalities)``.
        """
        self._retrieval_features = all_features
        self._retrieval_mask = all_mask

    def _retrieve_subgraph(
        self,
        item_idx: int,
        item_mask: np.ndarray,
        adjacency: sp.spmatrix,
    ) -> set[int]:
        """Retrieve anchors and expand into a subgraph for a query item.

        Args:
            item_idx: Index of the query item.
            item_mask: Binary mask of observed modalities for the query item.
            adjacency: Sparse item-item adjacency matrix.

        Returns:
            Set of node indices in the expanded subgraph.
        """
        mods = list(self._retrieval_features.keys())
        anchors: set[int] = set()
        for m_idx, m in enumerate(mods):
            if item_mask[m_idx] > 0:
                anchors.update(
                    anchor_retrieval(
                        item_idx,
                        m,
                        self._retrieval_features,
                        self._retrieval_mask,
                        top_k=self.top_k_anchors,
                    )
                )
        if not anchors:
            return {item_idx}
        return mage(
            adjacency,
            list(anchors),
            item_idx,
            self._retrieval_features,
            self._retrieval_mask,
            max_iters=self.mage_iters,
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
        adjacency: sp.spmatrix,
        index: torch.Tensor | None = None,
        training: bool = True,
    ) -> tuple:
        """Forward pass through GRE-MC.

        When :pyattr:`_retrieval_features` and :pyattr:`_retrieval_mask`
        are registered, the model performs modality-aware subgraph
        retrieval for each query item and jointly encodes the retrieved
        subgraph with the query node.  Otherwise it falls back to
        encoding the query node alone.

        Args:
            features: Dict of tensors, each with shape ``(B, d_m)``.
                These are the *query* item features for the current batch.
            mask: Binary indicator matrix of shape ``(B, M)``.
            adjacency: Sparse adjacency matrix for the item graph.
            index: Optional item indices of shape ``(B,)`` to index cached
                PE and to identify query items for retrieval.
            training: Whether to apply Gumbel noise in the codebook.

        Returns:
            Tuple of ``(predictions, routing_weights)`` where predictions is a
            dict of tensors with the same shapes as ``features``, and
            routing_weights has shape ``(B, codebook_size)``.
        """
        pe = self.pe_encoder(adjacency)
        batch_size = mask.shape[0]

        if (self._retrieval_features is not None
                and self._retrieval_mask is not None):
            if index is None:
                index = torch.arange(batch_size, device=mask.device)

            query_embeddings = []
            for b in range(batch_size):
                item_idx = int(index[b].item())
                item_mask_np = self._retrieval_mask[item_idx]

                # Retrieve subgraph and always include the query node.
                subgraph = self._retrieve_subgraph(
                    item_idx, item_mask_np, adjacency
                )
                subgraph = subgraph | {item_idx}
                node_list = sorted(subgraph)
                node_idx_t = torch.tensor(
                    node_list, device=mask.device, dtype=torch.long
                )

                # Gather tokens for the subgraph nodes.
                sub_features = {
                    m: torch.from_numpy(
                        self._retrieval_features[m][node_list]
                    )
                    .to(mask.device)
                    .float()
                    .unsqueeze(0)
                    for m in self._retrieval_features
                }
                sub_mask = (
                    torch.from_numpy(self._retrieval_mask[node_list])
                    .to(mask.device)
                    .float()
                    .unsqueeze(0)
                )
                sub_pe = pe[node_idx_t].to(mask.device).unsqueeze(0)

                # Single-item batch through the transformer.
                z = self.transformer(sub_features, sub_mask, sub_pe)
                query_embeddings.append(z)

            z = torch.cat(query_embeddings, dim=0)
        else:
            # Fallback: encode only the query node.
            pe_batch = pe[index] if index is not None else pe[:batch_size]
            z = self.transformer(features, mask, pe_batch)

        q, g = self.codebook(z, training=training)
        x_hat = self.decoder(q)
        return x_hat, g
