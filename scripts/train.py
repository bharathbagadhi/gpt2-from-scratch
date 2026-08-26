#!/usr/bin/env python3
"""Train GPT-2 from scratch, or fine-tune from OpenAI's weights.

Single device:
    python scripts/train.py --preset gpt2-nano --max_steps 500

Two GPUs with DistributedDataParallel:
    torchrun --standalone --nproc_per_node=2 scripts/train.py --strategy ddp

Two GPUs with FullyShardedDataParallel:
    torchrun --standalone --nproc_per_node=2 scripts/train.py --strategy fsdp

Fine-tune the released GPT-2 on Shakespeare:
    python scripts/train.py --init_from gpt2 --data_dir data/tinyshakespeare \\
        --block_size 256 --batch_size 4 --learning_rate 3e-5 --max_steps 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpt2.config import GPTConfig, TrainConfig  # noqa: E402
from gpt2.train import Trainer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    p.add_argument("--config", type=str, default=None,
                   help="JSON file of TrainConfig fields (see configs/). Any flag given "
                        "on the command line overrides the file.")

    g = p.add_argument_group("model")
    g.add_argument("--preset", default="gpt2",
                   choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "gpt2-nano"])
    g.add_argument("--n_layer", type=int, default=None)
    g.add_argument("--n_head", type=int, default=None)
    g.add_argument("--n_embd", type=int, default=None)
    g.add_argument("--dropout", type=float, default=0.0)
    g.add_argument("--vocab_size", type=int, default=None,
                   help="default 50304 = 50257 padded to a multiple of 64; the dead rows "
                        "cost nothing and the aligned matmul is measurably faster on CUDA. "
                        "Set this to your vocab size when using a char tokenizer.")

    g = p.add_argument_group("data")
    g.add_argument("--data_dir", default="data/tinyshakespeare")
    g.add_argument("--out_dir", default="out")

    g = p.add_argument_group("optimisation")
    g.add_argument("--batch_size", type=int, default=12)
    g.add_argument("--block_size", type=int, default=1024)
    g.add_argument("--grad_accum_steps", type=int, default=1)
    g.add_argument("--total_batch_size", type=int, default=None,
                   help="target tokens per optimiser step; derives grad_accum_steps")
    g.add_argument("--max_steps", type=int, default=5000)
    g.add_argument("--learning_rate", type=float, default=6e-4)
    g.add_argument("--min_lr_ratio", type=float, default=0.1)
    g.add_argument("--warmup_steps", type=int, default=100)
    g.add_argument("--weight_decay", type=float, default=0.1)
    g.add_argument("--beta1", type=float, default=0.9)
    g.add_argument("--beta2", type=float, default=0.95)
    g.add_argument("--grad_clip", type=float, default=1.0)

    g = p.add_argument_group("runtime")
    g.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    g.add_argument("--dtype", default="auto",
                   choices=["auto", "bfloat16", "float16", "float32"])
    g.add_argument("--compile", action="store_true")
    g.add_argument("--seed", type=int, default=1337)
    g.add_argument("--strategy", default="auto", choices=["auto", "single", "ddp", "fsdp"])
    g.add_argument("--init_from", default="scratch",
                   help="scratch | resume | gpt2 | gpt2-medium | gpt2-large | gpt2-xl")

    g = p.add_argument_group("logging")
    g.add_argument("--eval_interval", type=int, default=250)
    g.add_argument("--eval_iters", type=int, default=50)
    g.add_argument("--log_interval", type=int, default=10)
    g.add_argument("--checkpoint_interval", type=int, default=0)
    g.add_argument("--sample_interval", type=int, default=0)
    g.add_argument("--sample_prompt", default="\n")
    g.add_argument("--always_save_checkpoint", action="store_true")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Precedence: argparse defaults < --config file < explicit command-line flags.
    # argparse cannot distinguish "not given" from "given a value equal to the
    # default", so we read the flag names straight out of sys.argv. Anything the
    # user actually typed wins over the file; everything else the file supplies.
    if args.config:
        typed = {a.split("=", 1)[0].lstrip("-").replace("-", "_")
                 for a in sys.argv[1:] if a.startswith("--")}
        applied = []
        for key, value in TrainConfig.from_json(args.config).to_dict().items():
            if hasattr(args, key) and key not in typed:
                setattr(args, key, value)
                applied.append(key)
        print(f"loaded {args.config} ({len(applied)} settings; "
              f"{len(typed & set(vars(args))) - 1} overridden on the command line)")

    mcfg = GPTConfig.from_preset(args.preset)
    for k in ("n_layer", "n_head", "n_embd"):
        if getattr(args, k) is not None:
            setattr(mcfg, k, getattr(args, k))
    mcfg.dropout = args.dropout
    mcfg.block_size = max(mcfg.block_size, args.block_size)
    if args.vocab_size is not None:
        mcfg.vocab_size = args.vocab_size
    elif args.preset != "gpt2-nano":
        # Pad 50257 up to a multiple of 64. The extra rows can never be sampled
        # (no token maps to them) and their logits are driven to -inf by training,
        # but the aligned matmul is a free few percent on CUDA.
        mcfg.vocab_size = 50304
    mcfg.__post_init__()  # re-validate after the overrides

    tcfg = TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        block_size=args.block_size,
        grad_accum_steps=args.grad_accum_steps,
        total_batch_size=args.total_batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        min_lr_ratio=args.min_lr_ratio,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        grad_clip=args.grad_clip,
        device=args.device,
        dtype=args.dtype,
        compile=args.compile,
        seed=args.seed,
        strategy=args.strategy,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        sample_interval=args.sample_interval,
        sample_prompt=args.sample_prompt,
        always_save_checkpoint=args.always_save_checkpoint,
        init_from=args.init_from,
    )

    Trainer(mcfg, tcfg).train()


if __name__ == "__main__":
    main()
