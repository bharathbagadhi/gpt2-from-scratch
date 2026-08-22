"""The training loop.

Runs unchanged under three launchers:

    python scripts/train.py                                   # 1 device
    torchrun --nproc_per_node=2 scripts/train.py --strategy ddp
    torchrun --nproc_per_node=2 scripts/train.py --strategy fsdp

Things that are easy to get wrong and are handled explicitly here:

* **Gradient accumulation with DDP.** DDP all-reduces gradients in the backward
  pass. Doing that on every micro-step wastes bandwidth -- we only need the
  reduction on the last one. `no_sync()` suppresses it for the others.
* **Loss scaling under accumulation.** Cross-entropy already means over its
  batch, so summing N micro-step losses over-counts by N. Each micro-loss is
  divided by `grad_accum_steps` to recover the true mean.
* **float16 needs a GradScaler**, bfloat16 does not. The scaler is created as a
  no-op unless dtype is fp16.
* **Clip after unscale.** Gradient clipping on scaled fp16 gradients clips the
  wrong quantity.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import GPTConfig, TrainConfig
from .data import build_loaders
from .distributed import gather_state_dict, init_distributed, unwrap, wrap_model
from .model import GPT
from .utils import (
    autocast_ctx,
    device_type_of,
    enable_tf32,
    format_time,
    get_lr,
    human,
    pick_device,
    pick_dtype,
    set_seed,
)


class Trainer:
    def __init__(self, model_cfg: GPTConfig, train_cfg: TrainConfig) -> None:
        self.mcfg = model_cfg
        self.tcfg = train_cfg

        self.ctx = init_distributed(train_cfg.strategy, train_cfg.device)
        self.device = self.ctx.device if self.ctx.enabled else pick_device(train_cfg.device)
        self.dtype = pick_dtype(train_cfg.dtype, self.device)

        # Offset the seed per rank so ranks do not draw identical dropout masks.
        set_seed(train_cfg.seed + self.ctx.rank)
        if self.device.startswith("cuda"):
            enable_tf32()

        self.grad_accum = train_cfg.resolved_grad_accum(self.ctx.world_size)
        self.tokens_per_step = (
            train_cfg.batch_size * train_cfg.block_size * self.grad_accum * self.ctx.world_size
        )

        self.step = 0
        self.best_val = float("inf")
        self.out_dir = Path(train_cfg.out_dir)
        if self.ctx.is_master:
            self.out_dir.mkdir(parents=True, exist_ok=True)

        self._build_model()
        self._build_data()
        self._build_optimizer()

    # ----------------------------------------------------------------- setup
    def _log(self, *a) -> None:
        if self.ctx.is_master:
            print(*a, flush=True)

    def _build_model(self) -> None:
        init = self.tcfg.init_from
        if init == "scratch":
            model = GPT(self.mcfg)
            self._log(f"initialised GPT from scratch: {human(model.num_params())} params")
        elif init == "resume":
            ckpt = torch.load(self.out_dir / "ckpt.pt", map_location="cpu", weights_only=False)
            self.mcfg = GPTConfig.from_dict(ckpt["model_config"])
            model = GPT(self.mcfg)
            model.load_state_dict(ckpt["model"])
            self.step = ckpt.get("step", 0)
            self.best_val = ckpt.get("best_val", float("inf"))
            self._resume_optim_state = ckpt.get("optimizer")
            self._log(f"resumed from step {self.step} (best val {self.best_val:.4f})")
        elif init.startswith("gpt2"):
            from .pretrained import load_pretrained

            model = load_pretrained(init, dropout=self.mcfg.dropout)
            self.mcfg = model.config
            self._log(f"loaded pretrained {init}: {human(model.num_params())} params")
        else:
            raise ValueError(f"unknown init_from {init!r}")

        if self.tcfg.block_size < self.mcfg.block_size:
            model.crop_block_size(self.tcfg.block_size)
            self._log(f"cropped block_size to {self.tcfg.block_size}")

        model.to(self.device)
        self.raw_model = model

        if self.tcfg.compile:
            self._log("compiling model (first step will be slow)...")
            model = torch.compile(model)

        self.model = wrap_model(model, self.ctx, self.dtype)

    def _build_data(self) -> None:
        self.train_loader, self.val_loader = build_loaders(
            self.tcfg.data_dir,
            self.tcfg.batch_size,
            self.tcfg.block_size,
            device=self.device,
            rank=self.ctx.rank,
            world_size=self.ctx.world_size,
            seed=self.tcfg.seed,
        )

    def _build_optimizer(self) -> None:
        self.optimizer = self.raw_model.configure_optimizers(
            self.tcfg.weight_decay,
            self.tcfg.learning_rate,
            (self.tcfg.beta1, self.tcfg.beta2),
            device_type=device_type_of(self.device),
            verbose=self.ctx.is_master,
        )
        state = getattr(self, "_resume_optim_state", None)
        if state is not None:
            self.optimizer.load_state_dict(state)
            del self._resume_optim_state
        # enabled=False makes every scaler call a cheap no-op.
        self.scaler = torch.amp.GradScaler(
            device_type_of(self.device), enabled=(self.dtype == torch.float16)
        )

    # ------------------------------------------------------------- eval loop
    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        out = {}
        for split, loader in (("train", self.train_loader), ("val", self.val_loader)):
            if loader is None:
                continue
            losses = torch.zeros(self.tcfg.eval_iters, device=self.device)
            for i in range(self.tcfg.eval_iters):
                x, y = loader.next_batch()
                with autocast_ctx(self.device, self.dtype):
                    _, loss = self.model(x, y)
                losses[i] = loss.detach()
            mean = self.ctx.all_reduce_mean(losses.mean())
            out[split] = mean.item()
        self.model.train()
        return out

    # ------------------------------------------------------------ train step
    def _train_step(self) -> tuple[float, float]:
        """One optimiser step = `grad_accum` micro-steps. Returns (loss, grad_norm)."""
        self.optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(self.grad_accum):
            x, y = self.train_loader.next_batch()
            is_last = micro == self.grad_accum - 1

            # Suppress DDP's gradient all-reduce on every micro-step but the last.
            sync_ctx = (
                self.model.no_sync()
                if (not is_last and hasattr(self.model, "no_sync"))
                else _null()
            )
            with sync_ctx:
                with autocast_ctx(self.device, self.dtype):
                    _, loss = self.model(x, y)
                    # Undo the double-mean introduced by accumulation.
                    loss = loss / self.grad_accum
                accum_loss += loss.detach()
                self.scaler.scale(loss).backward()

        # Unscale before clipping, or we clip a scaled quantity (fp16 only).
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.tcfg.grad_clip
        ).item()

        lr = get_lr(
            self.step,
            base_lr=self.tcfg.learning_rate,
            warmup_steps=self.tcfg.warmup_steps,
            max_steps=self.tcfg.max_steps,
            min_lr_ratio=self.tcfg.min_lr_ratio,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self.scaler.step(self.optimizer)
        self.scaler.update()

        loss_t = self.ctx.all_reduce_mean(torch.as_tensor(accum_loss, device=self.device))
        return float(loss_t.item()), norm

    # ----------------------------------------------------------------- train
    def train(self) -> dict[str, float]:
        cfg = self.tcfg
        self._log(
            f"\ndevice={self.device}  dtype={str(self.dtype).split('.')[-1]}  "
            f"world_size={self.ctx.world_size}  strategy={self.ctx.strategy}\n"
            f"tokens/step = {self.tokens_per_step:,} "
            f"(B={cfg.batch_size} x T={cfg.block_size} x accum={self.grad_accum} "
            f"x ranks={self.ctx.world_size})\n"
            f"total budget = {human(self.tokens_per_step * cfg.max_steps)} tokens\n"
        )

        self.model.train()
        t_start = time.perf_counter()
        history: list[dict] = []
        last = time.perf_counter()

        while self.step < cfg.max_steps:
            # ---- periodic eval / checkpoint ----
            if self.step % cfg.eval_interval == 0 and self.val_loader is not None:
                m = self.evaluate()
                self._log(
                    f"step {self.step:5d} | train {m.get('train', float('nan')):.4f} "
                    f"| val {m.get('val', float('nan')):.4f}"
                )
                history.append({"step": self.step, **m})
                val = m.get("val", float("inf"))
                if val < self.best_val:
                    self.best_val = val
                    if self.step > 0:
                        self.save_checkpoint("ckpt.pt")
                elif cfg.always_save_checkpoint and self.step > 0:
                    self.save_checkpoint("ckpt.pt")

            if cfg.sample_interval and self.step % cfg.sample_interval == 0 and self.step > 0:
                self._sample()

            # ---- the actual step ----
            loss, norm = self._train_step()
            self.step += 1

            if self.step % cfg.log_interval == 0:
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                now = time.perf_counter()
                dt = (now - last) / cfg.log_interval
                last = now
                tps = self.tokens_per_step / dt
                lr_now = self.optimizer.param_groups[0]["lr"]
                self._log(
                    f"step {self.step:5d} | loss {loss:.4f} | lr {lr_now:.2e} "
                    f"| norm {norm:.2f} | {dt * 1000:6.0f} ms/step "
                    f"| {human(tps)} tok/s"
                )

            if cfg.checkpoint_interval and self.step % cfg.checkpoint_interval == 0:
                self.save_checkpoint(f"ckpt_step{self.step}.pt")

        # ---- final ----
        final = self.evaluate() if self.val_loader is not None else {}
        if final:
            self._log(
                f"\nfinal | train {final.get('train', float('nan')):.4f} "
                f"| val {final.get('val', float('nan')):.4f}"
            )
        self.save_checkpoint("ckpt_final.pt")
        self._log(f"done in {format_time(time.perf_counter() - t_start)}")
        self.ctx.cleanup()
        return {"history": history, **final}

    # ------------------------------------------------------------ side tasks
    def _sample(self) -> None:
        if not self.ctx.is_master:
            return
        try:
            from .tokenizer import BPETokenizer

            enc = BPETokenizer()
            ids = torch.tensor(
                [enc.encode(self.tcfg.sample_prompt) or [enc.eot_token]],
                dtype=torch.long,
                device=self.device,
            )
            out = unwrap(self.raw_model).generate(ids, 80, temperature=0.8, top_k=50)
            self._log("--- sample ---\n" + enc.decode(out[0].tolist()) + "\n--------------")
        except Exception as exc:  # sampling must never kill a training run
            self._log(f"(sampling skipped: {exc})")

    def save_checkpoint(self, name: str) -> Path | None:
        sd = gather_state_dict(self.model, self.ctx)
        if not self.ctx.is_master or sd is None:
            return None
        path = self.out_dir / name
        torch.save(
            {
                "model": sd,
                "optimizer": self.optimizer.state_dict(),
                "model_config": self.mcfg.to_dict(),
                "train_config": asdict(self.tcfg),
                "step": self.step,
                "best_val": self.best_val,
            },
            path,
        )
        self._log(f"saved {path}")
        return path


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def train_from_config(model_cfg: GPTConfig, train_cfg: TrainConfig) -> dict:
    return Trainer(model_cfg, train_cfg).train()
