"""Dataset preparation and loading.

Design decision worth defending in an interview: tokenised data is written once
to a flat `uint16` binary file and read back with `np.memmap`, not held in a
`torch.utils.data.Dataset`.

Why:
* GPT-2's vocabulary is 50,257, so a token fits in `uint16` -- 2 bytes/token
  instead of 8. A 10B-token corpus is 20 GB rather than 80 GB.
* `memmap` lets the OS page in only the windows we touch, so the corpus never
  has to fit in RAM.
* Language-model "examples" are arbitrary windows into one long stream, so an
  index-based `Dataset` adds bookkeeping and a collate step for nothing.

`DistributedTokenLoader` shards the stream by rank so that in a DDP run no two
GPUs ever see the same tokens in the same epoch.
"""

from __future__ import annotations

import os
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

DTYPE = np.uint16  # valid while vocab_size < 65536

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #
def download_tinyshakespeare(dest: str | Path) -> Path:
    """Fetch the 1.1 MB TinyShakespeare corpus (free, no auth, no API key)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"downloading tinyshakespeare -> {dest}")
    urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, dest)
    return dest


def write_bin(tokens: np.ndarray, path: str | Path) -> Path:
    """Write a token array to a flat uint16 .bin file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(tokens, dtype=DTYPE)
    if arr.ndim != 1:
        raise ValueError("expected a 1-D token stream")
    arr.tofile(path)
    return path


def read_bin(path: str | Path) -> np.ndarray:
    """Memory-map a .bin token file. Nothing is read until it is indexed."""
    return np.memmap(path, dtype=DTYPE, mode="r")


def prepare_text_dataset(
    text: str,
    out_dir: str | Path,
    tokenizer,
    val_fraction: float = 0.0005,
    min_val_tokens: int = 2048,
) -> dict[str, int]:
    """Tokenise a string and write train.bin / val.bin.

    The split is by position, not shuffled: the validation set is the *tail* of
    the corpus. Shuffling windows of a single continuous document leaks context
    across the split and gives you a validation loss that flatters you.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = np.array(tokenizer.encode(text), dtype=DTYPE)
    n_val = max(min_val_tokens, int(len(ids) * val_fraction))
    n_val = min(n_val, len(ids) // 10)  # never hand more than 10% to validation
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]

    write_bin(train_ids, out_dir / "train.bin")
    write_bin(val_ids, out_dir / "val.bin")

    meta = {"train_tokens": int(len(train_ids)), "val_tokens": int(len(val_ids))}
    (out_dir / "meta.json").write_text(
        f'{{"train_tokens": {meta["train_tokens"]}, '
        f'"val_tokens": {meta["val_tokens"]}, '
        f'"vocab_size": {tokenizer.vocab_size}}}\n'
    )
    print(
        f"wrote {meta['train_tokens']:,} train / {meta['val_tokens']:,} val tokens "
        f"to {out_dir}"
    )
    return meta


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
class TokenLoader:
    """Sequential batches of (x, y) next-token pairs from a .bin file.

    For a window of B*T+1 tokens, `x` is the first B*T and `y` is the same
    window shifted one to the left, so `y[i]` is the token that follows `x[i]`.
    One forward pass therefore supplies B*T supervised examples, not B.
    """

    def __init__(
        self,
        path: str | Path,
        batch_size: int,
        block_size: int,
        device: str = "cpu",
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = False,
        seed: int = 1337,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found -- run `python scripts/prepare_data.py` first"
            )
        self.tokens = read_bin(self.path)
        self.B, self.T = batch_size, block_size
        self.device = device
        self.rank, self.world_size = rank, world_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed + rank)

        self.stride = self.B * self.T * self.world_size
        if len(self.tokens) < self.stride + 1:
            raise ValueError(
                f"{self.path.name} has {len(self.tokens):,} tokens, too few for "
                f"batch_size*block_size*world_size = {self.stride:,}. "
                f"Lower --batch_size or --block_size."
            )
        self.reset()

    def __len__(self) -> int:
        """Number of full batches in one epoch, for this rank."""
        return (len(self.tokens) - 1) // self.stride

    def reset(self) -> None:
        # Each rank starts one shard along, so ranks never overlap.
        self.pos = self.B * self.T * self.rank
        self.epoch = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T
        if self.shuffle:
            hi = len(self.tokens) - (B * T + 1)
            start = int(self.rng.integers(0, max(1, hi)))
            buf = torch.from_numpy(
                self.tokens[start : start + B * T + 1].astype(np.int64)
            )
        else:
            if self.pos + B * T + 1 > len(self.tokens):
                self.pos = B * T * self.rank
                self.epoch += 1
            buf = torch.from_numpy(
                self.tokens[self.pos : self.pos + B * T + 1].astype(np.int64)
            )
            self.pos += self.stride

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        if self.device.startswith("cuda"):
            # Pinned + non_blocking overlaps the H2D copy with compute.
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        while True:
            yield self.next_batch()


def build_loaders(
    data_dir: str | Path,
    batch_size: int,
    block_size: int,
    device: str = "cpu",
    rank: int = 0,
    world_size: int = 1,
    seed: int = 1337,
) -> tuple[TokenLoader, TokenLoader | None]:
    """Train loader plus a val loader if val.bin exists."""
    data_dir = Path(data_dir)
    train = TokenLoader(
        data_dir / "train.bin", batch_size, block_size, device, rank, world_size, seed=seed
    )
    val_path = data_dir / "val.bin"
    val = None
    if val_path.exists():
        try:
            val = TokenLoader(
                val_path, batch_size, block_size, device, rank, world_size, seed=seed
            )
        except ValueError:
            # Validation split too small for this batch shape; skip rather than die.
            print(f"note: {val_path.name} too small for this batch shape, skipping eval")
    return train, val


def dataset_stats(data_dir: str | Path) -> dict[str, int]:
    data_dir = Path(data_dir)
    out = {}
    for split in ("train", "val"):
        p = data_dir / f"{split}.bin"
        if p.exists():
            out[f"{split}_tokens"] = os.path.getsize(p) // np.dtype(DTYPE).itemsize
    return out
