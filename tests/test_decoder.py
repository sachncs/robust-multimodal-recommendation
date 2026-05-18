import torch

from rmr.models.decoder import ModalityDecoders


def test_decoder_shapes():
    dec = ModalityDecoders(
        latent_dim=16,
        output_dims={"visual": 10, "text": 8},
    )
    q = torch.randn(5, 16)
    out = dec(q)
    assert out["visual"].shape == (5, 10)
    assert out["text"].shape == (5, 8)


def test_decoder_custom_hidden_dim():
    dec = ModalityDecoders(
        latent_dim=8,
        output_dims={"visual": 4, "text": 2},
        hidden_dim=32,
    )
    q = torch.randn(3, 8)
    out = dec(q)
    assert out["visual"].shape == (3, 4)
    assert out["text"].shape == (3, 2)


def test_decoder_single_modality():
    dec = ModalityDecoders(
        latent_dim=16,
        output_dims={"audio": 20},
    )
    q = torch.randn(7, 16)
    out = dec(q)
    assert out["audio"].shape == (7, 20)
