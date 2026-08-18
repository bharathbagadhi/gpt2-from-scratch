"""Configuration objects for model and training.

Kept as plain dataclasses (no hydra/omegaconf) so that every knob is
discoverable with `python -c "from gpt2.config import GPTConfig; help(GPTConfig)"`
and so the configs serialise cleanly into checkpoints.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class GPTConfig:
    """Architecture hyper-parameters.

    Defaults reproduce **GPT-2 small (124M)** exactly as released by OpenAI.

    The often-quoted "125M" figure counts the token + position embeddings
    twice (once as `wte`, once as the tied `lm_head`). The true number of
    *unique* parameters is 124,439,808 -- `GPT.num_params()` reports both.
    """

    block_size: int = 1024
    """Maximum context length in tokens."""

    vocab_size: int = 50257
    """GPT-2 BPE vocabulary: 50,000 merges + 256 bytes + 1 <|endoftext|>."""

    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    dropout: float = 0.0
    """0.0 is right for pretraining; raise to ~0.1 when fine-tuning on small data."""

    bias: bool = True
    """GPT-2 uses biases in Linear and LayerNorm. False is slightly faster."""

    # --- presets -------------------------------------------------------
    PRESETS: dict[str, dict[str, int]] = field(
        default_factory=lambda: {}, repr=False, compare=False
    )

    @classmethod
    def from_preset(cls, name: str, **overrides: Any) -> GPTConfig:
        """Build a config from a named GPT-2 size.

        >>> GPTConfig.from_preset("gpt2").n_layer
        12
        """
        presets = {
            # name          n_layer  n_head  n_embd
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M
            # A deliberately tiny model so that CI and laptops can run the
            # full training loop in seconds.
            "gpt2-nano": dict(
                n_layer=4, n_head=4, n_embd=128, block_size=128, vocab_size=50304
            ),
        }
        if name not in presets:
            raise KeyError(f"unknown preset {name!r}; choose from {sorted(presets)}")
        cfg = cls(**presets[name])
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise KeyError(f"unknown GPTConfig field {k!r}")
            setattr(cfg, k, v)
        return cfg

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("PRESETS", None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GPTConfig:
        known = {f.name for f in fields(cls)} - {"PRESETS"}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class TrainConfig:
    """Everything the training loop needs. Serialised into every checkpoint."""

    # --- data ---
    data_dir: str = "data/tinyshakespeare"
    out_dir: str = "out"

    # --- token budget -----------------------------------------------
    # The effective batch is  batch_size * block_size * grad_accum * world_size.
    # GPT-2 used 0.5M tokens/step; grad_accum lets a single small GPU emulate
    # that batch size at the cost of wall-clock time.
    batch_size: int = 12
    block_size: int = 1024
    grad_accum_steps: int = 1
    total_batch_size: int | None = None
    """If set (in tokens), grad_accum_steps is derived from it automatically."""

    # --- optimisation ---
    max_steps: int = 5000
    learning_rate: float = 6e-4
    min_lr_ratio: float = 0.1
    """Cosine decay floor, as a fraction of learning_rate. GPT-3 paper uses 0.1."""
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # --- runtime ---
    device: str = "auto"
    """'auto' | 'cuda' | 'mps' | 'cpu'."""
    dtype: str = "auto"
    """'auto' | 'bfloat16' | 'float16' | 'float32'."""
    compile: bool = False
    """torch.compile. Big speedup on CUDA, unsupported/slow elsewhere."""
    seed: int = 1337

    # --- distributed ---
    strategy: str = "auto"
    """'auto' | 'single' | 'ddp' | 'fsdp'. 'auto' picks ddp when torchrun is detected."""

    # --- logging / checkpointing ---
    eval_interval: int = 250
    eval_iters: int = 100
    log_interval: int = 10
    checkpoint_interval: int = 1000
    always_save_checkpoint: bool = False
    sample_interval: int = 0
    """If >0, generate a short sample every N steps so you can watch it learn."""
    sample_prompt: str = "\n"

    # --- init ---
    init_from: str = "scratch"
    """'scratch' | 'resume' | 'gpt2' | 'gpt2-medium' | 'gpt2-large' | 'gpt2-xl'."""

    def resolved_grad_accum(self, world_size: int = 1) -> int:
        """Derive grad_accum_steps from a target token budget, if one was given."""
        if self.total_batch_size is None:
            return self.grad_accum_steps
        per_step = self.batch_size * self.block_size * world_size
        if self.total_batch_size % per_step != 0:
            raise ValueError(
                f"total_batch_size ({self.total_batch_size}) must be divisible by "
                f"batch_size * block_size * world_size ({per_step})"
            )
        return self.total_batch_size // per_step

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        """Build from a dict, rejecting unknown keys.

        Unknown keys are an error rather than a silent ignore: a config file
        with `learnign_rate` should fail loudly, not train for six hours at the
        default learning rate. Keys beginning with `_` are treated as comments,
        which is how the files in `configs/` document themselves (JSON has no
        comment syntax).
        """
        d = {k: v for k, v in d.items() if not k.startswith("_")}
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown TrainConfig fields: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, path: str | Path) -> TrainConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")
