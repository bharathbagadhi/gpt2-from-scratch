"""Tests for the schedule, the config plumbing and a real end-to-end run."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gpt2.config import GPTConfig, TrainConfig
from gpt2.data import DTYPE, write_bin
from gpt2.pretrained import load_checkpoint
from gpt2.train import Trainer
from gpt2.utils import get_lr, pick_dtype


# ------------------------------------------------------------------ schedule
def test_lr_warms_up_linearly_from_near_zero():
    kw = dict(base_lr=1e-3, warmup_steps=10, max_steps=100)
    assert get_lr(0, **kw) == pytest.approx(1e-4)
    assert get_lr(4, **kw) == pytest.approx(5e-4)
    assert get_lr(9, **kw) == pytest.approx(1e-3)


def test_lr_peaks_at_end_of_warmup_then_decays():
    kw = dict(base_lr=1e-3, warmup_steps=10, max_steps=100)
    lrs = [get_lr(s, **kw) for s in range(100)]
    assert lrs.index(max(lrs)) == 9
    assert all(a >= b - 1e-12 for a, b in zip(lrs[9:], lrs[10:], strict=False)), "not monotonic after peak"


def test_lr_floor_is_min_lr_ratio():
    kw = dict(base_lr=1e-3, warmup_steps=10, max_steps=100, min_lr_ratio=0.1)
    assert get_lr(100, **kw) == pytest.approx(1e-4)
    assert get_lr(10_000, **kw) == pytest.approx(1e-4), "must clamp past max_steps"


def test_lr_handles_zero_warmup():
    assert get_lr(0, base_lr=1e-3, warmup_steps=0, max_steps=100) == pytest.approx(1e-3)


# -------------------------------------------------------------------- configs
def test_grad_accum_derived_from_token_budget():
    cfg = TrainConfig(batch_size=4, block_size=128, total_batch_size=4096)
    assert cfg.resolved_grad_accum(world_size=1) == 8
    assert cfg.resolved_grad_accum(world_size=2) == 4


def test_grad_accum_rejects_indivisible_budget():
    cfg = TrainConfig(batch_size=4, block_size=128, total_batch_size=5000)
    with pytest.raises(ValueError, match="divisible"):
        cfg.resolved_grad_accum()


def test_config_roundtrips_through_dict():
    cfg = GPTConfig(n_layer=6, n_head=6, n_embd=384)
    assert GPTConfig.from_dict(cfg.to_dict()) == cfg


def test_train_config_rejects_typos():
    with pytest.raises(KeyError, match="learnign_rate"):
        TrainConfig.from_dict({"learnign_rate": 1e-3})


def test_preset_lookup():
    assert GPTConfig.from_preset("gpt2").n_layer == 12
    assert GPTConfig.from_preset("gpt2-xl").n_embd == 1600
    with pytest.raises(KeyError):
        GPTConfig.from_preset("gpt2-enormous")


def test_dtype_selection_is_safe_on_cpu():
    assert pick_dtype("auto", "cpu") is torch.float32
    assert pick_dtype("bfloat16", "cpu") is torch.bfloat16


# --------------------------------------------------------------- integration
@pytest.fixture
def tiny_dataset(tmp_path):
    """A small repeating pattern. It is learnable, so the loss must actually
    fall -- a random stream would leave loss flat and the test would prove
    nothing."""
    rng = np.random.default_rng(0)
    motif = rng.integers(0, 97, size=64)
    stream = np.tile(motif, 400).astype(DTYPE)
    d = tmp_path / "data"
    write_bin(stream, d / "train.bin")
    write_bin(stream[:4096], d / "val.bin")
    return d


def test_end_to_end_training_run_reduces_loss(tiny_dataset, tmp_path):
    """Builds a Trainer, runs real steps, checkpoints, and reloads. This is the
    test that would catch a broken optimiser step, a mis-shaped batch, or a
    checkpoint that cannot be read back."""
    mcfg = GPTConfig(block_size=32, vocab_size=97, n_layer=2, n_head=2, n_embd=64)
    tcfg = TrainConfig(
        data_dir=str(tiny_dataset),
        out_dir=str(tmp_path / "out"),
        batch_size=4,
        block_size=32,
        max_steps=60,
        warmup_steps=5,
        learning_rate=3e-3,
        eval_interval=30,
        eval_iters=3,
        log_interval=1000,
        checkpoint_interval=0,
        device="cpu",
        dtype="float32",
        strategy="single",
    )

    trainer = Trainer(mcfg, tcfg)
    start = trainer.evaluate()["val"]
    result = trainer.train()
    end = result["val"]

    assert end < start, f"loss did not fall: {start:.3f} -> {end:.3f}"
    assert (tmp_path / "out" / "ckpt_final.pt").exists()

    # A checkpoint you cannot reload is not a checkpoint.
    reloaded = load_checkpoint(str(tmp_path / "out" / "ckpt_final.pt"), device="cpu")
    assert reloaded.config.n_layer == 2
    out = reloaded.generate(torch.zeros((1, 4), dtype=torch.long), 8)
    assert out.shape == (1, 12)


def test_gradient_accumulation_matches_one_big_batch(tiny_dataset, tmp_path):
    """Two micro-batches of 2 with accumulation must give the same gradient as
    one batch of 4. This is where the /grad_accum loss scaling is proved: drop
    it and the gradients come out 2x too large."""
    torch.manual_seed(0)
    mcfg = GPTConfig(block_size=16, vocab_size=97, n_layer=1, n_head=2, n_embd=32)
    from gpt2.model import GPT

    x = torch.randint(0, 97, (4, 16))
    y = torch.randint(0, 97, (4, 16))

    torch.manual_seed(0)
    big = GPT(mcfg)
    _, loss = big(x, y)
    loss.backward()
    ref = big.transformer.h[0].mlp.c_fc.weight.grad.clone()

    torch.manual_seed(0)
    small = GPT(mcfg)
    for i in range(2):
        _, micro_loss = small(x[i * 2 : (i + 1) * 2], y[i * 2 : (i + 1) * 2])
        (micro_loss / 2).backward()
    got = small.transformer.h[0].mlp.c_fc.weight.grad

    torch.testing.assert_close(ref, got, rtol=1e-4, atol=1e-6)


# ------------------------------------------------------------ shipped configs
def test_all_shipped_configs_parse():
    """Every JSON in configs/ must load. A config file that has drifted out of
    sync with TrainConfig is worse than no config file -- it fails six hours
    into someone's run, not at startup."""
    import glob
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    paths = sorted(glob.glob(str(root / "configs" / "*.json")))
    assert paths, "no configs found -- did the directory move?"
    for p in paths:
        cfg = TrainConfig.from_json(p)
        assert cfg.max_steps > 0, p
        assert cfg.batch_size > 0 and cfg.block_size > 0, p
        # A budget that is not divisible by the per-step token count would fail
        # only once training starts.
        cfg.resolved_grad_accum(world_size=8 if cfg.strategy == "ddp" else 1)


def test_json_comment_keys_are_ignored():
    cfg = TrainConfig.from_dict({"_comment": "hello", "max_steps": 7})
    assert cfg.max_steps == 7


def test_resume_restores_step_and_weights(tiny_dataset, tmp_path):
    """Resume must pick up the step counter and the weights, not silently
    restart from zero with a fresh model."""
    out = tmp_path / "out"
    mcfg = GPTConfig(block_size=32, vocab_size=97, n_layer=1, n_head=2, n_embd=32)
    base = dict(
        data_dir=str(tiny_dataset), out_dir=str(out), batch_size=4, block_size=32,
        warmup_steps=2, learning_rate=1e-3, eval_interval=10, eval_iters=2,
        log_interval=1000, device="cpu", dtype="float32", strategy="single",
        always_save_checkpoint=True,
    )

    first = Trainer(mcfg, TrainConfig(max_steps=20, **base))
    first.train()
    weights_before = first.raw_model.transformer.wte.weight.detach().clone()

    import shutil
    shutil.copy(out / "ckpt_final.pt", out / "ckpt.pt")

    second = Trainer(mcfg, TrainConfig(max_steps=30, init_from="resume", **base))
    assert second.step == 20, "resume did not restore the step counter"
    torch.testing.assert_close(
        second.raw_model.transformer.wte.weight, weights_before
    )
