"""Tests for sampling."""

from __future__ import annotations

import pytest
import torch

from gpt2.config import GPTConfig
from gpt2.model import GPT


@pytest.fixture
def model() -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(block_size=32, vocab_size=53, n_layer=2, n_head=2, n_embd=32))


def test_generate_appends_exactly_n_tokens(model):
    idx = torch.randint(0, 53, (2, 5))
    out = model.generate(idx, max_new_tokens=7)
    assert out.shape == (2, 12)
    # The prompt must be preserved verbatim at the front.
    torch.testing.assert_close(out[:, :5], idx)


def test_generate_respects_block_size(model):
    """A prompt longer than the context window must be cropped, not crash."""
    idx = torch.randint(0, 53, (1, model.config.block_size + 10))
    out = model.generate(idx, max_new_tokens=3)
    assert out.shape[1] == model.config.block_size + 13


def test_greedy_generation_is_deterministic(model):
    idx = torch.randint(0, 53, (1, 4))
    a = model.generate(idx, 10, temperature=0.0)
    b = model.generate(idx, 10, temperature=0.0)
    torch.testing.assert_close(a, b)


def test_top_k_restricts_the_support(model):
    """With top_k=1 sampling is argmax, so it must equal greedy decoding."""
    idx = torch.randint(0, 53, (1, 4))
    torch.manual_seed(0)
    a = model.generate(idx, 8, temperature=1.0, top_k=1)
    b = model.generate(idx, 8, temperature=0.0)
    torch.testing.assert_close(a, b)


def test_sampling_is_seed_reproducible(model):
    idx = torch.randint(0, 53, (1, 4))
    torch.manual_seed(42)
    a = model.generate(idx, 10, temperature=1.0, top_k=10)
    torch.manual_seed(42)
    b = model.generate(idx, 10, temperature=1.0, top_k=10)
    torch.testing.assert_close(a, b)


def test_generate_restores_training_mode(model):
    """generate() flips to eval internally; it must put the model back or the
    next training step silently runs without dropout."""
    model.train()
    model.generate(torch.randint(0, 53, (1, 3)), 2)
    assert model.training


def test_top_p_keeps_at_least_one_token(model):
    """Even an aggressive nucleus must leave the argmax in place, or softmax
    over all -inf produces NaN."""
    idx = torch.randint(0, 53, (1, 4))
    out = model.generate(idx, 6, temperature=1.0, top_p=0.01)
    assert out.shape == (1, 10)
    assert (out >= 0).all() and (out < 53).all()


def test_generated_ids_are_in_vocab(model):
    out = model.generate(torch.randint(0, 53, (3, 4)), 20, temperature=1.2, top_k=25)
    assert out.min() >= 0 and out.max() < 53
