"""Parity against HuggingFace's GPT-2.

These tests are the strongest claim the repo makes: load OpenAI's released
weights into *this* implementation and the outputs match the reference to
float tolerance. If `model.py` had a wrong scale factor, a transposed matrix,
a missing GELU approximation or an off-by-one in the positional embedding,
this is where it would show.

Skipped automatically when `transformers` is not installed or the weights
cannot be downloaded, so CI stays fast and offline-safe. Run them explicitly:

    pip install transformers
    pytest tests/test_pretrained.py -v
"""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip(
    "transformers", reason="install transformers to run parity tests"
)

from gpt2.pretrained import load_pretrained  # noqa: E402
from gpt2.tokenizer import BPETokenizer  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def pair():
    from transformers import GPT2LMHeadModel

    try:
        ours = load_pretrained("gpt2").eval()
        theirs = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    except Exception as exc:  # offline CI
        pytest.skip(f"could not download gpt2 weights: {exc}")
    return ours, theirs


def test_parameter_counts_agree(pair):
    ours, theirs = pair
    # HF counts the tied lm_head separately in some versions; compare unique.
    theirs_unique = sum(
        p.numel() for n, p in theirs.named_parameters() if n != "lm_head.weight"
    )
    assert ours.num_params() == theirs_unique


def test_logits_match_reference(pair):
    """The decisive test. Same tokens in, same logits out."""
    ours, theirs = pair
    idx = torch.randint(0, 50257, (2, 24))
    with torch.no_grad():
        ours_logits, _ = ours(idx, targets=idx)
        theirs_logits = theirs(idx).logits
    torch.testing.assert_close(ours_logits, theirs_logits, rtol=2e-4, atol=2e-4)


def test_loss_matches_reference(pair):
    ours, theirs = pair
    enc = BPETokenizer()
    idx = torch.tensor(
        [enc.encode("The Eiffel Tower is located in the city of Paris, France.")]
    )
    with torch.no_grad():
        _, our_loss = ours(idx, targets=idx)
        their_loss = theirs(idx, labels=idx).loss
    assert abs(our_loss.item() - their_loss.item()) < 1e-3


def test_pretrained_model_predicts_sensible_next_token(pair):
    """A behavioural check a human can read: greedy continuation of a factual
    prompt should be the obvious word."""
    ours, _ = pair
    enc = BPETokenizer()
    idx = torch.tensor([enc.encode("The capital of France is")])
    out = ours.generate(idx, max_new_tokens=3, temperature=0.0)
    assert "Paris" in enc.decode(out[0].tolist())


def test_pretrained_loss_is_far_better_than_random():
    """GPT-2 on ordinary English should sit well under ln(50257) ~ 10.8."""
    try:
        model = load_pretrained("gpt2").eval()
    except Exception as exc:
        pytest.skip(f"could not download gpt2 weights: {exc}")
    enc = BPETokenizer()
    text = (
        "Machine learning is a field of study in artificial intelligence "
        "concerned with the development of statistical algorithms that can "
        "learn from data and generalise to unseen examples."
    )
    idx = torch.tensor([enc.encode(text)])
    with torch.no_grad():
        _, loss = model(idx, targets=idx)
    assert loss.item() < 5.0, f"loss {loss.item():.2f} is suspiciously high"
