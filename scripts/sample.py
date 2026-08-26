#!/usr/bin/env python3
"""Generate text from a checkpoint or from OpenAI's released weights.

    python scripts/sample.py --init_from gpt2 --prompt "The capital of France is"
    python scripts/sample.py --ckpt out/ckpt_final.pt --prompt "ROMEO:" --num_samples 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpt2.pretrained import load_checkpoint, load_pretrained  # noqa: E402
from gpt2.tokenizer import BPETokenizer, CharTokenizer  # noqa: E402
from gpt2.utils import pick_device, set_seed  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", type=str, help="path to a checkpoint written by train.py")
    src.add_argument("--init_from", type=str,
                     help="gpt2 | gpt2-medium | gpt2-large | gpt2-xl")

    p.add_argument("--prompt", default="\n")
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8,
                   help="<1 sharpens, >1 flattens, 0 is greedy")
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto")
    p.add_argument("--char_vocab", type=str, default=None,
                   help="path to vocab.json if the checkpoint used a char tokenizer")
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)

    if args.ckpt:
        model = load_checkpoint(args.ckpt, device=device)
        print(f"loaded {args.ckpt}  ({model.num_params():,} params)")
    else:
        model = load_pretrained(args.init_from).to(device).eval()
        print(f"loaded pretrained {args.init_from}  ({model.num_params():,} params)")

    enc = CharTokenizer.load(args.char_vocab) if args.char_vocab else BPETokenizer()

    ids = enc.encode(args.prompt)
    if not ids:
        ids = [getattr(enc, "eot_token", 0)]
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for i in range(args.num_samples):
        t0 = time.perf_counter()
        out = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        dt = time.perf_counter() - t0
        print(f"\n{'=' * 70}\nsample {i + 1}/{args.num_samples}"
              f"   ({args.max_new_tokens / dt:.1f} tok/s)\n{'=' * 70}")
        print(enc.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
