"""GPT-2 implemented from scratch in PyTorch.

Every tensor operation in the forward pass is written out here -- there is no
`transformers` import anywhere in this file. The only reason `transformers`
appears in the project at all is `pretrained.py`, which ports OpenAI's released
weights into *this* implementation so the two can be checked against each other.

Architecture notes (where GPT-2 differs from the original 2017 Transformer):

1. **Pre-norm.** LayerNorm is applied to the *input* of each sub-block rather
   than to its output. This keeps a clean, un-normalised residual highway from
   the embeddings all the way to the final LayerNorm, which is what makes deep
   stacks trainable without a learning-rate warmup fight.
2. **Decoder only, causal mask.** Position `t` may attend to `[0, t]` only.
3. **Learned positional embeddings** (`wpe`), not sinusoids.
4. **GELU** activation, tanh approximation (that is what OpenAI shipped).
5. **Weight tying** between the token embedding and the output projection.
6. **Scaled residual init:** projections that write into the residual stream are
   initialised with std `0.02 / sqrt(2 * n_layer)`, so the residual stream's
   variance does not grow with depth.
"""

from __future__ import annotations

import inspect
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import GPTConfig


class LayerNorm(nn.Module):
    """LayerNorm with an *optional* bias.

    `torch.nn.LayerNorm` does not let you drop the bias while keeping the gain,
    and GPT-2 wants the bias while modern variants often do not. Twelve lines is
    cheaper than a config hack.
    """

    def __init__(self, ndim: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention.

    The three projections (query, key, value) are fused into a single
    `nn.Linear(n_embd, 3 * n_embd)` and split after the matmul. One large GEMM
    beats three small ones on every accelerator, and it matches the layout of
    OpenAI's released `c_attn` weight so the checkpoint port is a straight copy.

    Shapes, with B=batch, T=time, C=channels(n_embd), nh=n_head, hs=head_dim:

        x            (B, T, C)
        qkv          (B, T, 3C)      -> q, k, v each (B, T, C)
        reshaped     (B, nh, T, hs)
        attn scores  (B, nh, T, T)   <- the quadratic term
        out          (B, T, C)
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Tag for the scaled-residual init in GPT._init_weights.
        self.c_proj.NANOGPT_SCALE_INIT = True  # type: ignore[attr-defined]

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # PyTorch >= 2.0 ships a fused kernel (FlashAttention on CUDA) that never
        # materialises the (B, nh, T, T) score matrix. We prefer it, but keep the
        # explicit implementation below so the maths is visible and so the two can
        # be tested against each other (see tests/test_model.py).
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
                persistent=False,
            )

    def forward(self, x: torch.Tensor, use_flash: bool | None = None) -> torch.Tensor:
        B, T, C = x.size()
        flash = self.flash if use_flash is None else (use_flash and self.flash)

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, nh, T, hs); head becomes a batch dimension.
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if flash:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            # att[b,h,i,j] = <q_i, k_j> / sqrt(hs)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # Causal mask: -inf above the diagonal so softmax sends it to zero.
            mask = self._causal_mask(T, x.device)
            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)

        # Re-assemble heads side by side and project back into the residual stream.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        buf = getattr(self, "mask", None)
        if buf is not None and buf.size(-1) >= T:
            return buf[:, :, :T, :T]
        return torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)


class MLP(nn.Module):
    """Position-wise feed-forward network: C -> 4C -> GELU -> C.

    The 4x expansion is where most of the model's parameters live
    (8 * C^2 per layer against 4 * C^2 for attention).
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")  # what GPT-2 actually shipped
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = True  # type: ignore[attr-defined]
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """One transformer block.

    Note the *pre-norm* form: `x = x + f(norm(x))`, not `x = norm(x + f(x))`.
    The residual path is a clean identity, so gradients reach layer 0 undamped.
    Attention is where tokens exchange information ("reduce"); the MLP is where
    each token thinks about what it gathered ("map").
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """The full GPT-2 language model.

    Module names (`transformer.wte`, `transformer.h.0.attn.c_attn`, ...) match
    OpenAI's / HuggingFace's checkpoint layout on purpose: `from_pretrained`
    becomes a name-for-name copy instead of a translation table.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # token embedding
                wpe=nn.Embedding(config.block_size, config.n_embd),  # position embedding
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),  # final norm
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying. The embedding matrix maps token -> vector; the output
        # head maps vector -> token logits. Making them the same matrix saves
        # 38M parameters here (31% of the model) and measurably improves
        # perplexity: the two directions regularise each other.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Second pass for the scaled residual init (needs the depth).
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    # ------------------------------------------------------------------ init
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # --------------------------------------------------------------- forward
    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx:     (B, T) int64 token ids.
            targets: (B, T) int64 next-token ids, or None at inference.

        Returns:
            logits: (B, T, vocab_size) when training, (B, 1, vocab_size) when
                    `targets is None` -- we only need the last position to
                    sample, and computing the full head for a 1024-token prompt
                    wastes a 50257-wide matmul per position.
            loss:   scalar cross-entropy, or None.
        """
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.config.block_size}"
            )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)  # (B, T, C)
        pos_emb = self.transformer.wpe(pos)  # (T, C), broadcast over batch
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
            return logits, loss

        logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab)
        return logits, None

    # ----------------------------------------------------------- bookkeeping
    def num_params(self, non_embedding: bool = False) -> int:
        """Count parameters.

        Because `wte` and `lm_head` are tied they are one tensor, so a naive
        `sum(p.numel())` already counts them once -- 124,439,808 for GPT-2 small.
        Pass `non_embedding=True` to also subtract the position embeddings, the
        convention used when quoting "non-embedding parameters" in scaling laws.
        """
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def crop_block_size(self, block_size: int) -> None:
        """Shrink the context window in place (e.g. fine-tune GPT-2 at T=256)."""
        if block_size > self.config.block_size:
            raise ValueError("can only crop to a smaller block_size")
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "mask"):
                block.attn.mask = block.attn.mask[:, :, :block_size, :block_size]

    # ------------------------------------------------------------- optimiser
    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str = "cpu",
        verbose: bool = True,
    ) -> torch.optim.Optimizer:
        """AdamW with the standard two-group weight-decay split.

        Decay every tensor that participates in a matmul (weights, embeddings);
        do **not** decay 1-D tensors (biases, LayerNorm gains). Shrinking a
        LayerNorm gain toward zero is not regularisation, it is sabotage.
        """
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)

        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        if verbose:
            print(
                f"  decayed tensors: {len(decay):,} "
                f"({sum(p.numel() for p in decay):,} params)\n"
                f"  non-decayed    : {len(no_decay):,} "
                f"({sum(p.numel() for p in no_decay):,} params)"
            )

        # The fused AdamW kernel is a solid ~10% end-to-end win, CUDA only.
        fused_ok = (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
            and device_type == "cuda"
        )
        extra = {"fused": True} if fused_ok else {}
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)

    def estimate_mfu(self, tokens_per_iter: int, dt: float, flops_promised: float) -> float:
        """Model FLOPs Utilisation: what fraction of the GPU's peak we achieved.

        Uses the PaLM appendix-B estimate: 6*N FLOPs per token for the dense
        matmuls plus 12*L*H*Q*T for attention.
        """
        cfg = self.config
        N = self.num_params()
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.head_dim, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        achieved = flops_per_token * tokens_per_iter / dt
        return achieved / flops_promised

    # -------------------------------------------------------------- sampling
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend `idx` (B, T) by `max_new_tokens`.

        temperature: <1 sharpens the distribution, >1 flattens it, ->0 is greedy.
        top_k:       keep only the k most likely tokens before sampling.
        top_p:       nucleus sampling -- keep the smallest set whose cumulative
                     probability exceeds p. Composes with top_k.
        """
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                # Crop to the context window; GPT-2 has no memory beyond it.
                idx_cond = (
                    idx
                    if idx.size(1) <= self.config.block_size
                    else idx[:, -self.config.block_size :]
                )
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]

                if temperature <= 0:  # deterministic / greedy
                    idx_next = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k is not None:
                        k = min(top_k, logits.size(-1))
                        thresh = torch.topk(logits, k, dim=-1).values[:, [-1]]
                        logits = logits.masked_fill(logits < thresh, float("-inf"))
                    if top_p is not None and 0.0 < top_p < 1.0:
                        logits = self._top_p_filter(logits, top_p)
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)

                idx = torch.cat((idx, idx_next), dim=1)
                if eos_token_id is not None and (idx_next == eos_token_id).all():
                    break
            return idx
        finally:
            self.train(was_training)

    @staticmethod
    def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Drop everything past the point where cumulative mass exceeds top_p,
        # but always keep the single most likely token.
        remove = cum - F.softmax(sorted_logits, dim=-1) > top_p
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        return sorted_logits.scatter(-1, sorted_idx, sorted_logits)

    # ------------------------------------------------------------ pretrained
    @classmethod
    def from_pretrained(cls, model_type: str = "gpt2", **overrides) -> GPT:
        """Load OpenAI's released weights into this implementation.

        Thin re-export so `GPT.from_pretrained("gpt2")` works; the actual port
        lives in `gpt2.pretrained` to keep `transformers` out of this module.
        """
        from .pretrained import load_pretrained

        return load_pretrained(model_type, **overrides)
