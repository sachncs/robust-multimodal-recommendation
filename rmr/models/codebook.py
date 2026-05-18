"""Sparse-routing codebook with Gumbel-Softmax and Top-P selection."""

import torch
import torch.nn as nn
import torch.nn.functional as functional


class SparseRoutingCodebook(nn.Module):
    """Sparse-routing codebook with Gumbel-Softmax and Top-P selection.

    This module routes input representations through a learnable codebook
    using Gumbel-Softmax relaxation and restricts routing to the top-p
    entries for sparsity.

    Attributes:
        input_dim: Dimensionality of the input representations.
        codebook_size: Number of entries in the codebook.
        top_p: Number of top entries to retain per sample.
        tau: Temperature for the Gumbel-Softmax distribution.
    """

    def __init__(
        self,
        input_dim: int,
        codebook_size: int = 100,
        top_p: int = 4,
        tau: float = 0.5,
    ) -> None:
        """Initialize the sparse-routing codebook.

        Args:
            input_dim: Dimensionality of the input representations.
            codebook_size: Number of entries in the codebook.
            top_p: Number of top entries to retain for sparsity.
            tau: Temperature for the Gumbel-Softmax distribution.
        """
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.top_p = top_p
        self.tau = tau
        self.W = nn.Linear(input_dim, codebook_size)
        self.codebook = nn.Parameter(
            torch.randn(codebook_size, input_dim)
        )
        nn.init.xavier_uniform_(self.codebook)

    def forward(
        self, z: torch.Tensor, training: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route inputs through the codebook.

        Args:
            z: Input tensor of shape (batch_size, input_dim).
            training: Whether to use Gumbel noise during routing.

        Returns:
            A tuple containing:
                - quantized: Tensor of shape (batch_size, input_dim)
                  representing the quantized outputs.
                - routing_weights: Tensor of shape
                  (batch_size, codebook_size) containing the routing
                  weights before masking.
        """
        logits = self.W(z)
        if training:
            eps = -torch.log(
                -torch.log(torch.rand_like(logits) + 1e-10) + 1e-10
            )
            g = functional.softmax((logits + eps) / self.tau, dim=-1)
        else:
            g = functional.softmax(logits / self.tau, dim=-1)
        if self.top_p < self.codebook_size:
            top_vals, top_idx = torch.topk(g, self.top_p, dim=-1)
            mask = torch.zeros_like(g)
            mask.scatter_(1, top_idx, 1.0)
            g_masked = g * mask
            g_masked = g_masked / (
                g_masked.sum(dim=-1, keepdim=True) + 1e-10
            )
        else:
            g_masked = g
        q = g_masked @ self.codebook
        return q, g

    def usage_loss(self, g: torch.Tensor) -> torch.Tensor:
        """Compute the codebook usage loss.

        Measures the KL divergence between the empirical average routing
        distribution and a uniform distribution.

        Args:
            g: Routing weights tensor of shape (batch_size, codebook_size).

        Returns:
            Scalar tensor containing the usage loss.
        """
        bar_g = g.mean(dim=0)
        uniform = torch.ones_like(bar_g) / self.codebook_size
        kl = (
            bar_g
            * (
                torch.log(bar_g + 1e-10)
                - torch.log(uniform + 1e-10)
            )
        ).sum()
        return kl

    def load_loss(self, g: torch.Tensor) -> torch.Tensor:
        """Compute the codebook load-balancing loss.

        Penalizes imbalance by summing the squared mean routing weights
        scaled by the codebook size, matching the paper formula
        ``L_load = C * sum_e g_bar_e^2``.

        Args:
            g: Routing weights tensor of shape (batch_size, codebook_size).

        Returns:
            Scalar tensor containing the load loss.
        """
        bar_g = g.mean(dim=0)
        load = self.codebook_size * (bar_g ** 2).sum()
        return load
