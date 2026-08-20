"""Small helpers: device selection, dtype selection, seeding, LR schedule, timing."""

from __future__ import annotations

import math
import os
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch. Note this does not make CUDA fully
    deterministic -- for that you also need deterministic algorithms, which
    costs real throughput, so we do not enable it by default."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str = "auto") -> str:
    """Resolve 'auto' to the best available backend.

    Order: CUDA (real GPUs) > MPS (Apple Silicon) > CPU.
    """
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_type_of(device: str) -> str:
    return "cuda" if device.startswith("cuda") else device


def pick_dtype(requested: str, device: str) -> torch.dtype:
    """Resolve 'auto' to the widest fast dtype the device supports.

    bfloat16 is preferred over float16 because it has float32's exponent range,
    so training does not need a gradient scaler and cannot silently overflow.
    Ampere (A100, RTX 30xx) and newer support it; T4 and MPS do not.
    """
    if requested != "auto":
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
            requested
        ]
    if device.startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.startswith("cuda"):
        return torch.float16
    return torch.float32  # MPS/CPU autocast is not a reliable win


def autocast_ctx(device: str, dtype: torch.dtype):
    """Mixed-precision context, or a no-op on CPU/float32."""
    if dtype == torch.float32 or device == "cpu":
        return nullcontext()
    return torch.autocast(device_type=device_type_of(device), dtype=dtype)


def get_lr(step: int, *, base_lr: float, warmup_steps: int, max_steps: int,
           min_lr_ratio: float = 0.1) -> float:
    """Linear warmup then cosine decay to `min_lr_ratio * base_lr`.

    Warmup exists because Adam's second-moment estimate is garbage for the first
    few dozen steps; taking full-size steps on a garbage denominator is how you
    get a loss spike you never recover from.
    """
    min_lr = base_lr * min_lr_ratio
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (base_lr - min_lr)


def human(n: float) -> str:
    """1234567 -> '1.23M'."""
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


@contextmanager
def timed(device: str = "cpu") -> Iterator[list]:
    """Wall-clock timer that synchronises CUDA first.

    Without the sync you are timing kernel *launches*, not kernel execution,
    and every step looks like it takes 200 microseconds.
    """
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    box: list = []
    t0 = time.perf_counter()
    try:
        yield box
    finally:
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        box.append(time.perf_counter() - t0)


def enable_tf32() -> None:
    """Allow TF32 on Ampere+ matmuls: ~3x the FP32 throughput, and for training
    the reduced mantissa is not measurable in the loss curve."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def is_torchrun() -> bool:
    """True when launched under `torchrun` / `torch.distributed.run`."""
    return int(os.environ.get("RANK", -1)) != -1
