"""Per-modality decoder MLPs.

Each modality has its own MLP head. A learned [MASK] token replaces the
zero-multiplication leak from the legacy implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Decoders(nn.Module):
    """One MLP head per modality, with a learned [MASK] token per modality.

    The mask token is broadcast onto positions where the modality is missing
    before decoding. This is the standard masked-reconstruction practice
    (Devlin et al. 2019).
    """

    def __init__(
        self,
        latent_dim: int,
        dims: dict[str, int],
        hidden: int | None = None,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if not dims:
            raise ValueError("dims must not be empty")
        self.modalities = list(dims.keys())
        self.dims = dims
        if hidden is None:
            hidden = latent_dim
        self.heads = nn.ModuleDict()
        for name, dim in dims.items():
            self.heads[name] = nn.Sequential(
                nn.Linear(latent_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, dim),
            )
        self.mask_tokens = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(latent_dim)) for name in dims}
        )

    def forward(
        self,
        latent: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode per modality.

        Args:
            latent: ``(B, latent_dim)`` quantized code.
            mask: Optional ``(B, M)`` availability. Missing positions are
                augmented with the learned mask token before decoding.

        Returns
        -------
            Dict mapping modality name to ``(B, dim)`` reconstruction.
        """
        outputs: dict[str, torch.Tensor] = {}
        for idx, name in enumerate(self.modalities):
            effective = latent
            if mask is not None:
                present = mask[:, idx : idx + 1]
                missing = 1.0 - present
                token = self.mask_tokens[name].unsqueeze(0).expand_as(latent)
                effective = effective * present + token * missing
            outputs[name] = self.heads[name](effective)
        return outputs


__all__ = ["Decoders"]
