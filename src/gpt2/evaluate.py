"""Evaluation: perplexity, and the HellaSwag zero-shot benchmark.

Perplexity is `exp(cross_entropy)` -- the effective number of tokens the model
is choosing between at each position. A uniform model over GPT-2's vocabulary
scores 50,257; GPT-2 small scores roughly 30-40 on ordinary English.

HellaSwag is the standard sanity check for a 124M model. Each item is a context
plus four candidate endings; the model scores each ending by average token
log-likelihood and the argmax is its answer. Random is 25%. GPT-2 124M gets
about 29-30% -- barely above chance, which is the honest result and worth
reporting as such rather than quietly omitting.
"""

from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import GPT

HELLASWAG_VAL_URL = (
    "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/"
    "hellaswag_val.jsonl"
)


@torch.no_grad()
def perplexity(model: GPT, loader, iters: int = 100, device: str = "cpu") -> float:
    """Mean perplexity over `iters` batches."""
    model.eval()
    total = 0.0
    for _ in range(iters):
        x, y = loader.next_batch()
        _, loss = model(x, y)
        total += loss.item()
    model.train()
    return math.exp(total / iters)


# --------------------------------------------------------------------------- #
# HellaSwag
# --------------------------------------------------------------------------- #
def download_hellaswag(dest: str | Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"downloading hellaswag val -> {dest}")
        urllib.request.urlretrieve(HELLASWAG_VAL_URL, dest)
    return dest


def _render_example(example: dict, enc) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build (tokens, mask, label) for one HellaSwag item.

    `mask` is 1 on the *completion* tokens only -- the shared context must not
    contribute to the score, or every candidate gets the same free credit and
    longer contexts dominate.
    """
    ctx = example["ctx"]
    label = int(example["label"])
    ctx_tokens = enc.encode(ctx)

    rows, masks = [], []
    for ending in example["endings"]:
        end_tokens = enc.encode(" " + ending)  # leading space: GPT-2 BPE detail
        rows.append(ctx_tokens + end_tokens)
        masks.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

    max_len = max(len(r) for r in rows)
    tokens = torch.zeros(4, max_len, dtype=torch.long)
    mask = torch.zeros(4, max_len, dtype=torch.long)
    for i, (r, m) in enumerate(zip(rows, masks, strict=True)):
        tokens[i, : len(r)] = torch.tensor(r)
        mask[i, : len(m)] = torch.tensor(m)
    return tokens, mask, label


@torch.no_grad()
def hellaswag_accuracy(
    model: GPT,
    data_path: str | Path,
    device: str = "cpu",
    limit: int | None = None,
    enc=None,
) -> dict[str, float]:
    """Zero-shot HellaSwag accuracy.

    Reports two variants because the literature uses both:
      * `acc`      -- argmin of *summed* loss (favours short endings)
      * `acc_norm` -- argmin of *mean* per-token loss (length-normalised; this
                      is the number usually quoted, and the fairer one)
    """
    if enc is None:
        from .tokenizer import BPETokenizer

        enc = BPETokenizer()

    model.eval()
    n = n_correct = n_correct_norm = 0
    with open(data_path) as f:
        for line in f:
            if limit is not None and n >= limit:
                break
            tokens, mask, label = _render_example(json.loads(line), enc)
            if tokens.size(1) > model.config.block_size:
                continue
            tokens, mask = tokens.to(device), mask.to(device)

            logits, _ = model(tokens, targets=tokens)  # (4, T, V)
            # Shift so that position t predicts token t+1.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_tokens = tokens[:, 1:].contiguous()
            losses = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_tokens.view(-1),
                reduction="none",
            ).view(tokens.size(0), -1)

            shift_mask = mask[:, 1:].contiguous()
            masked = losses * shift_mask
            sum_loss = masked.sum(dim=1)
            avg_loss = sum_loss / shift_mask.sum(dim=1).clamp(min=1)

            n += 1
            n_correct += int(sum_loss.argmin().item() == label)
            n_correct_norm += int(avg_loss.argmin().item() == label)

    model.train()
    return {
        "n": n,
        "acc": n_correct / max(1, n),
        "acc_norm": n_correct_norm / max(1, n),
    }
