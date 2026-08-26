#!/usr/bin/env python3
"""Tokenise a corpus into train.bin / val.bin.

    python scripts/prepare_data.py --dataset tinyshakespeare
    python scripts/prepare_data.py --input my_corpus.txt --out_dir data/mine
    python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpt2.data import download_tinyshakespeare, prepare_text_dataset  # noqa: E402
from gpt2.tokenizer import BPETokenizer, CharTokenizer  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="tinyshakespeare", choices=["tinyshakespeare", "custom"])
    p.add_argument("--input", type=str, default=None, help="path to a .txt file (implies --dataset custom)")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--tokenizer", default="bpe", choices=["bpe", "char"])
    p.add_argument("--val_fraction", type=float, default=0.01)
    args = p.parse_args()

    if args.input:
        src = Path(args.input)
        if not src.exists():
            p.error(f"{src} does not exist")
        text = src.read_text(encoding="utf-8", errors="replace")
        out_dir = Path(args.out_dir or f"data/{src.stem}")
    else:
        raw = download_tinyshakespeare("data/raw/tinyshakespeare.txt")
        text = raw.read_text(encoding="utf-8")
        out_dir = Path(args.out_dir or f"data/tinyshakespeare{'_char' if args.tokenizer == 'char' else ''}")

    print(f"corpus: {len(text):,} characters")

    if args.tokenizer == "char":
        tok = CharTokenizer.fit(text)
        out_dir.mkdir(parents=True, exist_ok=True)
        tok.save(out_dir / "vocab.json")
        print(f"char vocab: {tok.vocab_size} symbols -> {out_dir / 'vocab.json'}")
    else:
        tok = BPETokenizer()
        print(f"tokenizer: {tok}")

    stats = prepare_text_dataset(text, out_dir, tok, val_fraction=args.val_fraction)
    ratio = len(text) / max(1, stats["train_tokens"] + stats["val_tokens"])
    print(f"compression: {ratio:.2f} characters per token")


if __name__ == "__main__":
    main()
