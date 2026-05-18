import torch

from rmr.models.codebook import SparseRoutingCodebook


def test_codebook_forward_shape():
    cb = SparseRoutingCodebook(
        input_dim=16,
        codebook_size=10,
        top_p=4,
        tau=0.5,
    )
    z = torch.randn(5, 16)
    q, g = cb(z, training=True)
    assert q.shape == (5, 16)
    assert g.shape == (5, 10)
    assert (g.sum(dim=1) - 1.0).abs().max() < 1e-4


def test_codebook_losses():
    cb = SparseRoutingCodebook(input_dim=8, codebook_size=5, top_p=2, tau=0.5)
    z = torch.randn(4, 8)
    q, g = cb(z, training=True)
    l_usage = cb.usage_loss(g)
    l_load = cb.load_loss(g)
    assert l_usage.item() >= 0
    assert l_load.item() >= 0


def test_top_p_masking_correctness():
    cb = SparseRoutingCodebook(input_dim=8, codebook_size=10, top_p=3, tau=0.5)
    z = torch.randn(6, 8)
    q, g = cb(z, training=False)
    # g is the raw softmax; top_p masking is applied inside forward to compute q,
    # but g returned is the unmasked raw softmax.
    # So we instead verify that q uses only top_p entries by inspecting the forward
    # more carefully: we can't directly test this on g. Let's re-run and capture
    # internals via hook, or test indirectly.
    # Actually, looking at code: q = g_masked @ self.codebook, and g is the raw softmax.
    # To verify top_p masking, we can monkey-patch or just check that with top_p=1
    # the quantized vector is exactly proportional to a single codebook row.
    cb2 = SparseRoutingCodebook(input_dim=4, codebook_size=5, top_p=1, tau=0.1)
    z2 = torch.randn(1, 4)
    q2, g2 = cb2(z2, training=False)
    # With top_p=1, only the max entry survives, so g_masked is a one-hot vector.
    # Therefore q2 should equal the codebook row with the highest logit.
    logits = cb2.W(z2)
    best_idx = logits.argmax(dim=-1).item()
    expected = cb2.codebook[best_idx]
    torch.testing.assert_close(q2.squeeze(0), expected, atol=1e-4, rtol=1e-3)


def test_training_false_no_gumbel_noise():
    cb = SparseRoutingCodebook(input_dim=8, codebook_size=5, top_p=2, tau=0.5)
    z = torch.randn(3, 8)
    torch.manual_seed(42)
    q1, g1 = cb(z, training=False)
    torch.manual_seed(99)
    q2, g2 = cb(z, training=False)
    # Without Gumbel noise, output should be deterministic regardless of seed
    torch.testing.assert_close(q1, q2, atol=1e-6, rtol=0)
    torch.testing.assert_close(g1, g2, atol=1e-6, rtol=0)


def test_gradient_flow():
    cb = SparseRoutingCodebook(input_dim=8, codebook_size=5, top_p=2, tau=0.5)
    z = torch.randn(2, 8, requires_grad=True)
    q, g = cb(z, training=True)
    loss = q.sum() + cb.usage_loss(g) + cb.load_loss(g)
    loss.backward()
    assert z.grad is not None
    assert cb.codebook.grad is not None
    assert cb.W.weight.grad is not None


def test_load_loss_includes_codebook_size():
    cb = SparseRoutingCodebook(input_dim=4, codebook_size=10, top_p=2, tau=0.5)
    z = torch.randn(1, 4)
    q, g = cb(z, training=False)
    l_load = cb.load_loss(g)
    bar_g = g.mean(dim=0)
    expected = 10.0 * (bar_g ** 2).sum()
    torch.testing.assert_close(l_load, expected, atol=1e-6, rtol=0)
