#!/usr/bin/env python3
"""Micro-benchmark: tokens/sec and peak memory for a given configuration.

Useful for the README table and for showing you can reason about throughput
rather than just wait for a loss curve.

    python scripts/benchmark.py --preset gpt2 --batch_size 4 --block_size 512
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpt2.config import GPTConfig  # noqa: E402
from gpt2.model import GPT  # noqa: E402
from gpt2.utils import autocast_ctx, human, pick_device, pick_dtype  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default="gpt2")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)

    cfg = GPTConfig.from_preset(args.preset)
    cfg.block_size = max(cfg.block_size, args.block_size)
    model = GPT(cfg).to(device)
    if args.compile:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    B, T = args.batch_size, args.block_size
    x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    y = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    print(f"preset={args.preset}  params={human(GPT(cfg).num_params())}  "
          f"device={device}  dtype={str(dtype).split('.')[-1]}  B={B} T={T}")

    def one_step() -> None:
        opt.zero_grad(set_to_none=True)
        with autocast_ctx(device, dtype):
            _, loss = model(x, y)
        loss.backward()
        opt.step()

    for _ in range(args.warmup):
        one_step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(args.steps):
        one_step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / args.steps

    print(f"  {dt * 1000:8.1f} ms / step")
    print(f"  {human(B * T / dt):>8} tokens / sec")
    if device.startswith("cuda"):
        print(f"  {torch.cuda.max_memory_allocated() / 1e9:8.2f} GB peak allocated")


if __name__ == "__main__":
    main()
