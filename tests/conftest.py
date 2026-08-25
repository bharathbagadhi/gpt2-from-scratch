"""Shared fixtures.

`bpe` is a session-scoped fixture rather than a bare `BPETokenizer()` call so
that tests degrade to a skip on a machine with no network. tiktoken fetches the
GPT-2 merge table from the internet the first time it is used and caches it;
sandboxed CI runners and offline laptops would otherwise show a wall of
connection errors that look like real failures.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def bpe():
    from gpt2.tokenizer import BPETokenizer

    try:
        return BPETokenizer()
    except Exception as exc:  # no network, or no tiktoken cache
        pytest.skip(f"GPT-2 BPE vocabulary unavailable offline: {type(exc).__name__}")
