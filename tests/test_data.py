"""Tests for the tokenizers and the data pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gpt2.data import DTYPE, TokenLoader, prepare_text_dataset, read_bin, write_bin
from gpt2.tokenizer import CharTokenizer

SAMPLE = "To be, or not to be, that is the question.\n" * 200


# ------------------------------------------------------------------ tokenizer
def test_char_tokenizer_roundtrip():
    tok = CharTokenizer.fit(SAMPLE)
    assert tok.decode(tok.encode(SAMPLE)) == SAMPLE


def test_char_tokenizer_skips_unknown_characters():
    tok = CharTokenizer.fit("abc")
    assert tok.decode(tok.encode("abcXYZ")) == "abc"


def test_char_tokenizer_save_load(tmp_path):
    tok = CharTokenizer.fit(SAMPLE)
    p = tmp_path / "vocab.json"
    tok.save(p)
    assert CharTokenizer.load(p).chars == tok.chars


def test_bpe_roundtrip_and_vocab_size(bpe):
    assert bpe.vocab_size == 50257
    assert bpe.eot_token == 50256
    # Byte-level BPE has no <unk>: em dashes and emoji must survive intact.
    text = "Hello world! GPT-2 uses byte-level BPE — even emoji 🎯 round-trip."
    assert bpe.decode(bpe.encode(text)) == text


def test_bpe_compresses_english(bpe):
    """Byte-level BPE should average roughly 4 characters per token on English.
    A big drift here means the wrong encoding was loaded."""
    text = "The quick brown fox jumps over the lazy dog. " * 50
    ratio = len(text) / len(bpe.encode(text))
    assert 3.0 < ratio < 6.0, f"{ratio:.2f} chars/token looks wrong"


# ----------------------------------------------------------------------- bins
def test_bin_roundtrip(tmp_path):
    toks = np.arange(1000, dtype=DTYPE)
    p = write_bin(toks, tmp_path / "t.bin")
    np.testing.assert_array_equal(np.asarray(read_bin(p)), toks)


def test_uint16_is_enough_for_gpt2_vocab():
    """Guard rail: the 2-bytes-per-token storage choice is only valid while the
    vocabulary fits in uint16."""
    assert np.iinfo(DTYPE).max >= 50304


def test_prepare_dataset_writes_split(tmp_path):
    tok = CharTokenizer.fit(SAMPLE)
    stats = prepare_text_dataset(SAMPLE, tmp_path, tok, val_fraction=0.1, min_val_tokens=64)
    assert (tmp_path / "train.bin").exists() and (tmp_path / "val.bin").exists()
    assert stats["train_tokens"] > stats["val_tokens"] > 0
    # Nothing lost, nothing duplicated.
    assert stats["train_tokens"] + stats["val_tokens"] == len(tok.encode(SAMPLE))


# --------------------------------------------------------------------- loader
@pytest.fixture
def bin_path(tmp_path):
    return write_bin(np.arange(5000, dtype=DTYPE), tmp_path / "train.bin")


def test_targets_are_inputs_shifted_by_one(bin_path):
    """The central contract of next-token prediction."""
    loader = TokenLoader(bin_path, batch_size=2, block_size=8)
    x, y = loader.next_batch()
    assert x.shape == (2, 8) and y.shape == (2, 8)
    torch.testing.assert_close(x[:, 1:], y[:, :-1])


def test_loader_advances_and_wraps(bin_path):
    loader = TokenLoader(bin_path, batch_size=2, block_size=8)
    first, _ = loader.next_batch()
    second, _ = loader.next_batch()
    assert not torch.equal(first, second)
    # Drain past the end of the file; it must wrap rather than raise.
    for _ in range(len(loader) + 5):
        loader.next_batch()
    assert loader.epoch >= 1


def test_ranks_see_disjoint_data(bin_path):
    """In DDP two ranks must not train on the same tokens in the same step."""
    a = TokenLoader(bin_path, 2, 8, rank=0, world_size=2)
    b = TokenLoader(bin_path, 2, 8, rank=1, world_size=2)
    xa, _ = a.next_batch()
    xb, _ = b.next_batch()
    assert set(xa.flatten().tolist()).isdisjoint(set(xb.flatten().tolist()))


def test_loader_reports_useful_error_when_file_too_small(tmp_path):
    p = write_bin(np.arange(10, dtype=DTYPE), tmp_path / "tiny.bin")
    with pytest.raises(ValueError, match="too few"):
        TokenLoader(p, batch_size=8, block_size=64)


def test_loader_reports_useful_error_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        TokenLoader(tmp_path / "nope.bin", 2, 8)
