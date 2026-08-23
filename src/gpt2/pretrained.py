"""Port OpenAI's released GPT-2 weights into this implementation.

This is the file that proves the model in `model.py` is really GPT-2 and not
merely GPT-2-shaped: copy the official weights in, and the loss and the
generations match HuggingFace's `GPT2LMHeadModel` to within float noise
(`tests/test_pretrained.py` asserts this when `transformers` is installed).

The one wrinkle: the original GPT-2 was written in TensorFlow using `Conv1D`,
whose weight is stored **transposed** relative to `nn.Linear`. HuggingFace kept
that layout for checkpoint compatibility, so four tensors per block need a
`.t()` on the way in. Every other tensor is a straight copy, because the module
names in `model.py` were chosen to match.
"""

from __future__ import annotations

from typing import Any

import torch

from .config import GPTConfig
from .model import GPT

# HF stores these as Conv1D (in, out); nn.Linear wants (out, in).
_TRANSPOSED_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)

_ARCH: dict[str, dict[str, int]] = {
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
}


def load_pretrained(model_type: str = "gpt2", **overrides: Any) -> GPT:
    """Return a :class:`GPT` holding OpenAI's weights for `model_type`.

    Requires `transformers` (used purely as a weight downloader; not a single
    layer of theirs runs in the forward pass).
    """
    if model_type not in _ARCH:
        raise ValueError(f"unknown model_type {model_type!r}; choose from {sorted(_ARCH)}")
    try:
        from transformers import GPT2LMHeadModel
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "loading pretrained weights needs the transformers package "
            "(as a downloader only):  pip install transformers"
        ) from exc

    cfg = GPTConfig(vocab_size=50257, block_size=1024, bias=True, **_ARCH[model_type])
    for k, v in overrides.items():
        setattr(cfg, k, v)

    model = GPT(cfg)
    sd = model.state_dict()
    # `.attn.mask` is a derived buffer, not a parameter; skip it.
    keys = [k for k in sd if not k.endswith(".attn.mask")]

    hf = GPT2LMHeadModel.from_pretrained(model_type)
    hf_sd = hf.state_dict()
    hf_keys = [
        k
        for k in hf_sd
        if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")
    ]

    # `lm_head.weight` is tied to `wte.weight`, so it may or may not be listed.
    ours = set(keys) - {"lm_head.weight"}
    theirs = set(hf_keys) - {"lm_head.weight"}
    if ours != theirs:
        missing, extra = sorted(theirs - ours), sorted(ours - theirs)
        raise RuntimeError(
            f"state dict mismatch porting {model_type}\n"
            f"  in HF but not ours: {missing[:8]}\n"
            f"  in ours but not HF: {extra[:8]}"
        )

    with torch.no_grad():
        for k in theirs:
            src = hf_sd[k]
            if k.endswith(_TRANSPOSED_SUFFIXES):
                src = src.t()
            if src.shape != sd[k].shape:
                raise RuntimeError(f"shape mismatch for {k}: {src.shape} vs {sd[k].shape}")
            sd[k].copy_(src)

    model.load_state_dict(sd, strict=False)
    return model


def load_checkpoint(path: str, device: str = "cpu", dropout: float | None = None) -> GPT:
    """Rebuild a :class:`GPT` from a checkpoint written by :class:`Trainer`."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = GPTConfig.from_dict(ckpt["model_config"])
    if dropout is not None:
        cfg.dropout = dropout
    model = GPT(cfg)

    sd = ckpt["model"]
    # torch.compile prefixes every key with `_orig_mod.`; strip it.
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model
