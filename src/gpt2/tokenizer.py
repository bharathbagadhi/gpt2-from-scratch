"""Tokenizers.

Two backends behind one interface:

* :class:`BPETokenizer` -- GPT-2's real byte-level BPE via `tiktoken`. 50,257
  tokens, ~4 characters each. This is what you want for anything real.
* :class:`CharTokenizer` -- one token per character, vocabulary built from the
  corpus. Useless for transfer learning, but it makes a from-scratch run on
  1 MB of Shakespeare converge in minutes on a laptop, which is exactly what
  you want while you are still debugging the training loop.

Both satisfy the same protocol, so `train.py` never branches on which is in use.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    vocab_size: int
    eot_token: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: Iterable[int]) -> str: ...


class BPETokenizer:
    """GPT-2 byte-level BPE (`tiktoken`'s `gpt2` encoding).

    Byte-level means every possible byte string is encodable -- there is no
    `<unk>` token and no way to feed it text it cannot represent.
    """

    def __init__(self, encoding_name: str = "gpt2") -> None:
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "BPETokenizer needs tiktoken:  pip install tiktoken"
            ) from exc
        self._enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self._enc.n_vocab  # 50257
        self.eot_token = self._enc.eot_token  # 50256, <|endoftext|>

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def encode_with_special(self, text: str) -> list[int]:
        """Encode allowing `<|endoftext|>` to appear literally in the text."""
        return self._enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, ids: Iterable[int]) -> str:
        return self._enc.decode(list(ids))

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={self.vocab_size})"


class CharTokenizer:
    """Character-level tokenizer with a vocabulary fitted to a corpus."""

    def __init__(self, chars: list[str]) -> None:
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.eot_token = 0

    @classmethod
    def fit(cls, text: str) -> CharTokenizer:
        return cls(sorted(set(text)))

    def encode(self, text: str) -> list[int]:
        # Unknown characters are skipped rather than crashing -- a char vocab
        # fitted on Shakespeare will not contain the emoji in your test prompt.
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos.get(int(i), "") for i in ids)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"chars": self.chars}, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        return cls(json.loads(Path(path).read_text())["chars"])

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"


def get_tokenizer(kind: str = "bpe", **kwargs) -> Tokenizer:
    """Factory. `kind` is 'bpe' or 'char'."""
    if kind == "bpe":
        return BPETokenizer(**kwargs)
    if kind == "char":
        path = kwargs.get("path")
        if path is None:
            raise ValueError("char tokenizer needs path= to a saved vocab, or use .fit()")
        return CharTokenizer.load(path)
    raise ValueError(f"unknown tokenizer kind {kind!r}; use 'bpe' or 'char'")
