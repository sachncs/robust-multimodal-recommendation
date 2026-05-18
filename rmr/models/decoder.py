"""Per-modality decoder modules for multi-modal representation learning."""


import torch
import torch.nn as nn


class ModalityDecoders(nn.Module):
    """Per-modality MLP decoders mapping latent codes back to feature space.

    This module maintains a separate decoder network for each modality.

    Attributes:
        decoders: ModuleDict mapping modality names to decoder networks.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dims: dict[str, int],
        hidden_dim: int = None,
    ) -> None:
        """Initialize the modality decoders.

        Args:
            latent_dim: Dimensionality of the latent input code.
            output_dims: Dictionary mapping modality names to their output
                dimensionalities.
            hidden_dim: Hidden layer dimension. Defaults to latent_dim if
                not specified.
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = latent_dim
        self.decoders = nn.ModuleDict()
        for name, dim in output_dims.items():
            self.decoders[name] = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim),
            )

    def forward(self, q: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode a latent code into per-modality reconstructions.

        Args:
            q: Latent tensor of shape (batch_size, latent_dim).

        Returns:
            Dictionary mapping each modality name to a reconstructed
            tensor of shape (batch_size, output_dim).
        """
        return {
            name: decoder(q)
            for name, decoder in self.decoders.items()
        }
