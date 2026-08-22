"""Distributed training: DDP and FSDP, behind one small interface.

The whole point of this module is that `train.py` should not contain a single
`if ddp:` branch. It asks for a :class:`DistContext`, wraps the model once, and
otherwise writes single-GPU code.

**DDP (DistributedDataParallel)** -- every rank holds a *full* replica of the
model, optimiser state and gradients. Each rank does a forward/backward on its
own shard of the batch, then gradients are all-reduced (averaged) so every
replica takes an identical step. Memory per GPU is unchanged; you buy throughput.
Correct choice while the model still fits comfortably on one GPU -- which
GPT-2 124M does, several times over.

**FSDP (FullyShardedDataParallel)** -- parameters, gradients *and* optimiser
state are sharded across ranks. Before a layer runs, its parameters are
all-gathered; afterwards they are freed again. Memory per GPU drops roughly by
`1/world_size`; you pay extra communication. The right tool when the model does
*not* fit, i.e. well above 124M.

Including FSDP here at 124M is deliberately a demonstration of the mechanism,
not a recommendation -- for this model size DDP is faster, and the code says so.

Memory intuition for AdamW at 124M in mixed precision:
    params fp32   0.50 GB
    grads  fp32   0.50 GB
    Adam m, v     1.00 GB
    ------------------------
    ~2.0 GB before activations. Comfortable on a free Colab T4 (16 GB).
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass
class DistContext:
    """Rank/device bookkeeping. Constructed once at process start."""

    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: str
    strategy: str  # 'single' | 'ddp' | 'fsdp'

    @property
    def is_master(self) -> bool:
        """Only rank 0 prints, logs and writes checkpoints."""
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def all_reduce_mean(self, t: torch.Tensor) -> torch.Tensor:
        """Average a scalar (e.g. the loss) across ranks, for honest logging."""
        if not self.enabled:
            return t
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        return t

    def cleanup(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


def init_distributed(strategy: str = "auto", device: str = "auto") -> DistContext:
    """Initialise the process group if we were launched under torchrun.

    Run single-process and this is a no-op that returns a world_size=1 context,
    so the same script works with `python` and with `torchrun`.
    """
    from .utils import pick_device

    ddp_run = int(os.environ.get("RANK", -1)) != -1

    if not ddp_run:
        if strategy in ("ddp", "fsdp"):
            raise RuntimeError(
                f"strategy={strategy!r} requires torchrun, e.g.\n"
                f"  torchrun --standalone --nproc_per_node=2 scripts/train.py ..."
            )
        return DistContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=pick_device(device),
            strategy="single",
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dev = f"cuda:{local_rank}"
    else:
        dev = "cpu"

    resolved = strategy if strategy != "auto" else "ddp"
    return DistContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=dev,
        strategy=resolved,
    )


def wrap_model(model: nn.Module, ctx: DistContext, mixed_precision_dtype=None) -> nn.Module:
    """Wrap `model` for the chosen strategy. Returns the model unchanged when single-process."""
    if not ctx.enabled or ctx.strategy == "single":
        return model
    if ctx.strategy == "ddp":
        return _wrap_ddp(model, ctx)
    if ctx.strategy == "fsdp":
        return _wrap_fsdp(model, ctx, mixed_precision_dtype)
    raise ValueError(f"unknown strategy {ctx.strategy!r}")


def _wrap_ddp(model: nn.Module, ctx: DistContext) -> nn.Module:
    from torch.nn.parallel import DistributedDataParallel as DDP

    device_ids = [ctx.local_rank] if ctx.device.startswith("cuda") else None
    return DDP(model, device_ids=device_ids)


def _wrap_fsdp(model: nn.Module, ctx: DistContext, mp_dtype) -> nn.Module:
    """Shard the model by transformer Block.

    `transformer_auto_wrap_policy` on `Block` is the standard recipe: it makes
    each block its own FSDP unit, so the all-gather granularity matches the
    compute granularity and communication overlaps with the next block's maths.
    Wrapping at a finer grain thrashes; wrapping the whole model as one unit
    defeats the purpose.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    from .model import Block

    # FSDP has no CPU path: it needs a device handle to shard onto. Fail here
    # with something actionable rather than deep inside _init_device_handle.
    if not ctx.device.startswith("cuda"):
        raise RuntimeError(
            "FSDP requires CUDA GPUs -- PyTorch has no CPU sharding backend.\n"
            f"  detected device: {ctx.device}\n"
            "  use --strategy ddp instead (it works on CPU with the gloo backend), "
            "or run this on a multi-GPU machine."
        )

    policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={Block}
    )
    mp = None
    if mp_dtype is not None and mp_dtype != torch.float32:
        mp = MixedPrecision(
            param_dtype=mp_dtype,
            reduce_dtype=mp_dtype,
            buffer_dtype=mp_dtype,
        )
    return FSDP(
        model,
        auto_wrap_policy=policy,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=ctx.local_rank if ctx.device.startswith("cuda") else None,
        use_orig_params=True,  # required for torch.compile and for param groups
    )


def unwrap(model: nn.Module) -> nn.Module:
    """Get the raw GPT back out of a DDP/FSDP wrapper (for .generate, .config)."""
    return getattr(model, "module", model)


def gather_state_dict(model: nn.Module, ctx: DistContext) -> dict | None:
    """Collect a full, unsharded state dict on rank 0 for checkpointing.

    Under FSDP each rank holds only a slice, so a plain `state_dict()` would
    save a fragment. `FullStateDictConfig(rank0_only=True, offload_to_cpu=True)`
    reassembles the whole thing on rank 0 without every rank allocating a full
    copy of the model in GPU memory.
    """
    if not ctx.enabled or ctx.strategy != "fsdp":
        return unwrap(model).state_dict() if ctx.is_master else None

    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        sd = model.state_dict()
    return sd if ctx.is_master else None
