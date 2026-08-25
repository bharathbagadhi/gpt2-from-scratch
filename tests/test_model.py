"""Tests for the model architecture.

The interesting ones are `test_attention_is_causal` (a real behavioural test of
the mask, not a shape check) and `test_flash_matches_manual` (the fused kernel
and the hand-written attention must agree).
"""

from __future__ import annotations

import math

import pytest
import torch

from gpt2.config import GPTConfig
from gpt2.model import GPT, CausalSelfAttention


@pytest.fixture
def tiny_config() -> GPTConfig:
    return GPTConfig(block_size=32, vocab_size=101, n_layer=2, n_head=2, n_embd=32)


@pytest.fixture
def tiny_model(tiny_config: GPTConfig) -> GPT:
    torch.manual_seed(0)
    return GPT(tiny_config)


# --------------------------------------------------------------------- shapes
def test_forward_shapes_with_targets(tiny_model, tiny_config):
    B, T = 2, 16
    idx = torch.randint(0, tiny_config.vocab_size, (B, T))
    logits, loss = tiny_model(idx, targets=idx)
    assert logits.shape == (B, T, tiny_config.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_forward_returns_last_position_only_at_inference(tiny_model, tiny_config):
    """Without targets we only need the final position, so the head is not run
    across the whole sequence -- a real speedup for long prompts."""
    idx = torch.randint(0, tiny_config.vocab_size, (2, 16))
    logits, loss = tiny_model(idx)
    assert logits.shape == (2, 1, tiny_config.vocab_size)
    assert loss is None


def test_rejects_sequence_longer_than_block_size(tiny_model, tiny_config):
    idx = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.block_size + 1))
    with pytest.raises(ValueError, match="exceeds block_size"):
        tiny_model(idx)


def test_config_rejects_indivisible_head_dim():
    with pytest.raises(ValueError, match="divisible"):
        GPTConfig(n_embd=100, n_head=12)


# ------------------------------------------------------------------ causality
def test_attention_is_causal(tiny_model, tiny_config):
    """Changing a *future* token must not change the logits at an earlier
    position. If the mask were wrong the model would be reading ahead, the loss
    would look wonderful, and generation would be gibberish."""
    tiny_model.eval()
    torch.manual_seed(1)
    idx = torch.randint(0, tiny_config.vocab_size, (1, 12))

    with torch.no_grad():
        logits_a, _ = tiny_model(idx, targets=idx)
        perturbed = idx.clone()
        perturbed[0, 8] = (perturbed[0, 8] + 1) % tiny_config.vocab_size
        logits_b, _ = tiny_model(perturbed, targets=perturbed)

    # Positions 0..7 precede the edit and must be bit-comparable.
    torch.testing.assert_close(logits_a[:, :8], logits_b[:, :8], rtol=1e-5, atol=1e-6)
    # Position 8 itself sees the change, so it must differ.
    assert not torch.allclose(logits_a[:, 8], logits_b[:, 8])


def test_flash_matches_manual_attention(tiny_config):
    """The fused SDPA kernel and the explicit softmax(QK^T/sqrt(d))V must agree."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(tiny_config).eval()
    if not attn.flash:
        pytest.skip("this torch build has no scaled_dot_product_attention")
    x = torch.randn(2, 16, tiny_config.n_embd)
    with torch.no_grad():
        fast = attn(x, use_flash=True)
        slow = attn(x, use_flash=False)
    torch.testing.assert_close(fast, slow, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------- parameters
def test_gpt2_small_parameter_count():
    """124,439,808 unique parameters. The familiar '125M' double-counts the tied
    embedding; this test pins the real number so a refactor cannot quietly
    change the architecture."""
    model = GPT(GPTConfig())
    assert model.num_params() == 124_439_808
    # Non-embedding count subtracts the 1024x768 position table.
    assert model.num_params(non_embedding=True) == 124_439_808 - 1024 * 768


def test_weight_tying_shares_one_tensor():
    model = GPT(GPTConfig(block_size=8, vocab_size=50, n_layer=1, n_head=1, n_embd=8))
    assert model.transformer.wte.weight is model.lm_head.weight
    model.lm_head.weight.data.fill_(0.123)
    assert torch.allclose(model.transformer.wte.weight, torch.full_like(
        model.transformer.wte.weight, 0.123))


def test_residual_projections_use_scaled_init():
    """std should be 0.02/sqrt(2*n_layer), not 0.02, for tensors that write into
    the residual stream."""
    n_layer = 12
    model = GPT(GPTConfig(n_layer=n_layer, n_head=4, n_embd=128, block_size=16))
    expected = 0.02 / math.sqrt(2 * n_layer)
    std = model.transformer.h[0].mlp.c_proj.weight.std().item()
    assert expected * 0.85 < std < expected * 1.15, f"got std={std}, expected ~{expected}"


def test_initial_loss_is_near_uniform(tiny_config):
    """At init the model knows nothing, so cross-entropy should be about
    ln(vocab_size). A wildly different value means the init is broken."""
    torch.manual_seed(0)
    model = GPT(tiny_config)
    idx = torch.randint(0, tiny_config.vocab_size, (4, 16))
    _, loss = model(idx, targets=idx)
    expected = math.log(tiny_config.vocab_size)
    assert abs(loss.item() - expected) < 0.5, f"loss {loss.item()} vs ln(V) {expected}"


# ------------------------------------------------------------------ optimiser
def test_optimizer_splits_decay_groups(tiny_model):
    opt = tiny_model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), verbose=False)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == 0.1
    assert opt.param_groups[1]["weight_decay"] == 0.0
    # Every non-decayed tensor must be 1-D (biases, LayerNorm gains).
    assert all(p.dim() < 2 for p in opt.param_groups[1]["params"])
    assert all(p.dim() >= 2 for p in opt.param_groups[0]["params"])


def test_crop_block_size(tiny_config):
    model = GPT(tiny_config)
    model.crop_block_size(16)
    assert model.config.block_size == 16
    assert model.transformer.wpe.weight.shape[0] == 16
    idx = torch.randint(0, tiny_config.vocab_size, (1, 16))
    logits, _ = model(idx, targets=idx)
    assert logits.shape[1] == 16


# ------------------------------------------------------------------- learning
def test_model_can_overfit_a_single_batch(tiny_config):
    """The end-to-end smoke test that matters: given one batch and enough steps,
    loss must collapse toward zero. If it does not, something in the
    forward/backward/optimiser path is disconnected."""
    torch.manual_seed(0)
    model = GPT(tiny_config)
    idx = torch.randint(0, tiny_config.vocab_size, (2, 12))
    opt = model.configure_optimizers(0.0, 3e-3, (0.9, 0.95), verbose=False)

    _, first = model(idx, targets=idx)
    for _ in range(220):
        opt.zero_grad(set_to_none=True)
        _, loss = model(idx, targets=idx)
        loss.backward()
        opt.step()

    assert loss.item() < first.item() * 0.1, (
        f"failed to overfit: {first.item():.3f} -> {loss.item():.3f}"
    )
    assert loss.item() < 0.5


def test_all_parameters_receive_gradients(tiny_config):
    """A parameter with no gradient is dead weight -- usually a sign of a layer
    built but never wired into the forward pass."""
    torch.manual_seed(0)
    model = GPT(tiny_config)
    idx = torch.randint(0, tiny_config.vocab_size, (2, 16))
    _, loss = model(idx, targets=idx)
    loss.backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None]
    assert not dead, f"parameters with no gradient: {dead}"
