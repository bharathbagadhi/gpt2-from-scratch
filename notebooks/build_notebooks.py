#!/usr/bin/env python3
"""Generate the two Colab notebooks from plain Python source.

Notebooks are terrible to review in a pull request: the JSON diff hides the
change and stored outputs make every commit noisy. Keeping the source here and
generating the .ipynb means the notebooks stay in version control as a
reviewable file, and `python notebooks/build_notebooks.py` rebuilds them.

    python notebooks/build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


SETUP = """
# Colab setup. Skip the clone if you are running this locally from the repo.
import os, sys

if not os.path.exists("gpt2-from-scratch"):
    !git clone -q https://github.com/BharathBagadhi/gpt2-from-scratch.git
%cd gpt2-from-scratch
!pip install -q -e . 2>/dev/null

sys.path.insert(0, "src")

import torch
print("torch", torch.__version__)
print("device:", "cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
"""


# ===================================================================== nb 1
NB1 = notebook([
    md("""
# Training GPT-2 from scratch

This notebook trains a GPT-2-architecture language model from random initialisation on
TinyShakespeare and watches it go from noise to English.

Everything here runs on **Colab's free T4** (or on CPU, more slowly). Nothing costs money
and no API key is needed.

**Runtime → Change runtime type → T4 GPU** before you start.
"""),
    code(SETUP),

    md("""
## 1. The data

TinyShakespeare is 1.1 MB of the complete works, concatenated. Small enough to download in a
second, large enough that a small model learns real structure from it.

We tokenise it two ways:
- **char** — one token per character, 65-symbol vocabulary. Converges fast; ideal for
  watching the training loop work.
- **BPE** — GPT-2's real 50,257-token byte-level BPE. What you need for anything transferable.
"""),
    code("""
!python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char
!python scripts/prepare_data.py --dataset tinyshakespeare
"""),

    md("""
### What a tokenizer actually does

Byte-level BPE compresses English to roughly 4 characters per token. Because it operates on
*bytes*, every possible input is encodable — there is no `<unk>` token and no text it cannot
represent.
"""),
    code("""
from gpt2.tokenizer import BPETokenizer, CharTokenizer

bpe = BPETokenizer()
text = "To be, or not to be, that is the question."

ids = bpe.encode(text)
print(f"vocab size : {bpe.vocab_size:,}")
print(f"text       : {text}")
print(f"tokens     : {ids}")
print(f"pieces     : {[bpe.decode([i]) for i in ids]}")
print(f"compression: {len(text) / len(ids):.2f} chars/token")
print(f"roundtrip  : {bpe.decode(ids) == text}")
"""),

    md("""
### Input and target are the same stream, shifted by one

This is the whole of the language-modelling objective. For a window of `B*T+1` tokens, `x` is
the first `B*T` and `y` is the same window shifted left by one, so `y[i]` is the token that
follows `x[i]`. One forward pass therefore supplies `B*T` supervised examples, not `B` — this
is why language models are so sample-efficient per unit of compute.
"""),
    code("""
from gpt2.data import TokenLoader

char_tok = CharTokenizer.load("data/tinyshakespeare_char/vocab.json")
loader = TokenLoader("data/tinyshakespeare_char/train.bin", batch_size=2, block_size=16)
x, y = loader.next_batch()

print("x[0]:", repr(char_tok.decode(x[0].tolist())))
print("y[0]:", repr(char_tok.decode(y[0].tolist())))
print()
for i in range(6):
    ctx = char_tok.decode(x[0, :i+1].tolist())
    tgt = char_tok.decode([y[0, i].item()])
    print(f"  given {ctx!r:<20} predict {tgt!r}")
"""),

    md("""
## 2. The model

`GPTConfig` holds the architecture. The defaults are GPT-2 small exactly; here we build a
smaller one so it trains in minutes rather than hours.
"""),
    code("""
from gpt2 import GPT, GPTConfig

full = GPT(GPTConfig())
print(f"GPT-2 small : {full.num_params():,} parameters")
print()

# Where they live.
groups = {
    "token embedding (tied with lm_head)": full.transformer.wte.weight.numel(),
    "position embedding": full.transformer.wpe.weight.numel(),
    "attention": sum(p.numel() for n, p in full.named_parameters() if ".attn." in n),
    "mlp": sum(p.numel() for n, p in full.named_parameters() if ".mlp." in n),
}
for name, n in groups.items():
    print(f"  {name:<38} {n:>12,}  ({n / full.num_params():5.1%})")

del full
"""),

    md("""
### Weight tying, and why the loss starts at ln(V)

`wte` (token → vector) and `lm_head` (vector → logits) are the *same* matrix. That saves 38.6M
parameters and improves perplexity, because the two directions regularise each other.

At initialisation the model knows nothing, so it should assign roughly uniform probability to
every token — cross-entropy `≈ ln(vocab_size)`. If your loss starts anywhere else, the
initialisation is broken, and you have found out in one second instead of after an hour of
training.
"""),
    code("""
import math

cfg = GPTConfig(block_size=128, vocab_size=char_tok.vocab_size,
                n_layer=6, n_head=6, n_embd=192)
model = GPT(cfg)

print("wte is lm_head:", model.transformer.wte.weight is model.lm_head.weight)
print(f"parameters    : {model.num_params():,}")

x, y = TokenLoader("data/tinyshakespeare_char/train.bin", 4, 128).next_batch()
_, loss = model(x, y)
print(f"\\ninitial loss  : {loss.item():.4f}")
print(f"ln(vocab_size): {math.log(cfg.vocab_size):.4f}   <- they should match")
"""),

    md("""
### What it generates before training

Uniform noise over the character vocabulary. Worth looking at once so the "after" is meaningful.
"""),
    code("""
import torch

prompt = torch.tensor([char_tok.encode("ROMEO:")], dtype=torch.long)
print(char_tok.decode(model.generate(prompt, 200, temperature=1.0)[0].tolist()))
"""),

    md("""
## 3. Train it

On a T4 this is about 2 minutes. On CPU, roughly 12.

Watch the loss: it drops fast at first (the model learns character *frequency* — that `e` and
space are common), then more slowly as it learns spelling, then word boundaries, then the
speaker-name-colon-newline structure of a play.
"""),
    code("""
!python scripts/train.py \\
    --preset gpt2-nano --n_layer 6 --n_head 6 --n_embd 192 \\
    --data_dir data/tinyshakespeare_char --vocab_size 65 \\
    --batch_size 32 --block_size 128 \\
    --max_steps 2500 --warmup_steps 100 --learning_rate 2e-3 \\
    --eval_interval 250 --eval_iters 25 --log_interval 100 \\
    --out_dir out/demo
"""),

    md("""
## 4. What it generates now
"""),
    code("""
from gpt2.pretrained import load_checkpoint

trained = load_checkpoint("out/demo/ckpt_final.pt", device="cpu")

for temp in (0.5, 0.8, 1.2):
    out = trained.generate(prompt, 300, temperature=temp, top_k=40)
    print(f"\\n{'=' * 70}\\ntemperature = {temp}\\n{'=' * 70}")
    print(char_tok.decode(out[0].tolist()))
"""),

    md("""
**Temperature** rescales the logits before the softmax. Below 1 it sharpens the distribution
(more repetitive, more confident); above 1 it flattens it (more varied, more mistakes). At 0
it is pure argmax and the model loops almost immediately.

## 5. Measure it

Perplexity is `exp(cross_entropy)` — the effective number of characters the model is choosing
between at each position. A uniform model over this 65-symbol vocabulary would score 65.

The gap between train and validation loss is the honest read on overfitting. On 1 MB of text
a larger model would show a much wider gap.
"""),
    code("""
from gpt2.data import build_loaders

train_loader, val_loader = build_loaders("data/tinyshakespeare_char", 16, 128)

@torch.no_grad()
def mean_loss(m, loader, iters=25):
    m.eval()
    return sum(m(*loader.next_batch())[1].item() for _ in range(iters)) / iters

tr = mean_loss(trained, train_loader)
va = mean_loss(trained, val_loader)
print(f"train loss : {tr:.4f}  (perplexity {math.exp(tr):5.2f})")
print(f"val   loss : {va:.4f}  (perplexity {math.exp(va):5.2f})")
print(f"uniform    : {math.log(65):.4f}  (perplexity {65:5.2f})   <- where we started")
print(f"\\ntrain/val gap: {va - tr:.4f}")
"""),

    md("""
### The training curve

Re-run the sweep across the checkpoints the training loop saved, or just plot the losses
printed above against the run's log. Here we plot the eval history recorded during training.
"""),
    code("""
import matplotlib.pyplot as plt

# The Trainer returns its eval history; re-run in-process to capture it.
from gpt2.config import GPTConfig, TrainConfig
from gpt2.train import Trainer

mcfg = GPTConfig(block_size=128, vocab_size=65, n_layer=4, n_head=4, n_embd=128)
tcfg = TrainConfig(
    data_dir="data/tinyshakespeare_char", out_dir="out/curve",
    batch_size=32, block_size=128, max_steps=600, warmup_steps=50,
    learning_rate=3e-3, eval_interval=50, eval_iters=20, log_interval=200,
)
hist = Trainer(mcfg, tcfg).train()["history"]

steps = [h["step"] for h in hist]
plt.figure(figsize=(7, 4))
plt.plot(steps, [h["train"] for h in hist], label="train", marker="o", ms=3)
plt.plot(steps, [h["val"] for h in hist], label="val", marker="s", ms=3)
plt.axhline(math.log(65), ls="--", c="gray", lw=1, label="uniform, ln(65)")
plt.xlabel("step"); plt.ylabel("cross-entropy loss")
plt.title("Training a 4-layer GPT on TinyShakespeare")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
"""),

    md("""
## 6. Scaling up

The same script trains the full 124M model. On a free T4:

```bash
python scripts/train.py --preset gpt2 \\
    --data_dir data/tinyshakespeare \\
    --batch_size 4 --block_size 512 --total_batch_size 65536 \\
    --max_steps 3000 --learning_rate 6e-4
```

`--total_batch_size` is in **tokens**; gradient accumulation is derived from it, so the
optimisation is identical on one T4 or eight A100s — only wall-clock time differs.

With more than one GPU:

```bash
torchrun --standalone --nproc_per_node=8 scripts/train.py --strategy ddp
```

Next: `02_pretrained_and_finetune.ipynb`, which loads OpenAI's actual GPT-2 weights into this
implementation.
"""),
])


# ===================================================================== nb 2
NB2 = notebook([
    md("""
# Pretrained GPT-2: loading, probing, and fine-tuning

The previous notebook trained a small model from scratch. This one does something more
interesting: it loads **OpenAI's released GPT-2 weights into this from-scratch
implementation** and shows the two are the same model.

Then it fine-tunes that 124M model on Shakespeare — on a free T4, in a few minutes.
"""),
    code(SETUP),
    code('!pip install -q transformers'),

    md("""
## 1. Load OpenAI's weights into our model

`transformers` is used here purely as a weight *downloader*. None of its layers run in the
forward pass — every tensor operation comes from `src/gpt2/model.py`.

The one wrinkle: the original GPT-2 was written in TensorFlow using `Conv1D`, whose weight is
stored transposed relative to `nn.Linear`. Four tensors per block need a `.t()` on the way in.
Everything else is a straight name-for-name copy, because the module names in `model.py` were
chosen to match the checkpoint.
"""),
    code("""
from gpt2.pretrained import load_pretrained
from gpt2.tokenizer import BPETokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_pretrained("gpt2").to(device).eval()
enc = BPETokenizer()

print(f"loaded GPT-2 small: {model.num_params():,} parameters")
"""),

    md("""
## 2. Prove it is really GPT-2

Same tokens into both implementations; the logits must match to float tolerance. This is the
test that would fail on a wrong scale factor, a transposed matrix, a missing GELU
approximation, or an off-by-one in the positional embedding.
"""),
    code("""
from transformers import GPT2LMHeadModel

reference = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
idx = torch.randint(0, 50257, (2, 32), device=device)

with torch.no_grad():
    ours, _ = model(idx, targets=idx)
    theirs = reference(idx).logits

diff = (ours - theirs).abs().max().item()
print(f"max |ours - huggingface| = {diff:.3e}")
assert diff < 1e-3
print("identical to float tolerance")
"""),

    md("""
## 3. What it knows

Greedy decoding (`temperature=0`) shows the model's single most confident continuation, which
is the clearest way to see what it has actually memorised.
"""),
    code("""
prompts = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The theory of relativity was developed by",
    "In 1969, humans first landed on",
]

for p in prompts:
    idx = torch.tensor([enc.encode(p)], device=device)
    out = model.generate(idx, max_new_tokens=12, temperature=0.0)
    print(f"{p!r}\\n  -> {enc.decode(out[0].tolist())}\\n")
"""),

    md("""
### Look inside the distribution

Rather than only sampling, inspect what the model believes. This is a more honest picture of
a 124M model than a cherry-picked generation.
"""),
    code("""
import torch.nn.functional as F

prompt = "The capital of France is"
idx = torch.tensor([enc.encode(prompt)], device=device)
with torch.no_grad():
    logits, _ = model(idx)

probs = F.softmax(logits[0, -1], dim=-1)
top = torch.topk(probs, 10)
print(f"{prompt!r} -> next token:\\n")
for p, i in zip(top.values.tolist(), top.indices.tolist()):
    bar = "#" * int(p * 60)
    print(f"  {enc.decode([i])!r:<14} {p:6.2%}  {bar}")
"""),

    md("""
### Sampling knobs

`temperature`, `top_k` and `top_p` all shape the same distribution differently. Worth seeing
side by side.
"""),
    code("""
prompt = "The most important thing about machine learning is"
idx = torch.tensor([enc.encode(prompt)], device=device)

settings = [
    ("greedy",            dict(temperature=0.0)),
    ("temp 0.7, top-k 40", dict(temperature=0.7, top_k=40)),
    ("temp 1.0, top-p 0.9", dict(temperature=1.0, top_p=0.9)),
    ("temp 1.4 (chaotic)", dict(temperature=1.4, top_k=100)),
]

for name, kw in settings:
    torch.manual_seed(0)
    out = model.generate(idx, max_new_tokens=50, **kw)
    print(f"--- {name} ---\\n{enc.decode(out[0].tolist())}\\n")
"""),

    md("""
## 4. Where GPT-2 124M actually lands

HellaSwag is the standard sanity benchmark: a context plus four candidate endings, scored by
average token log-likelihood. Random is 25%. GPT-2 124M gets about **29-30%** — barely above
chance.

That is the honest number, and reporting it matters more than hiding it. 124M parameters is
small; the interesting result is that it is above chance at all.
"""),
    code("""
from gpt2.evaluate import download_hellaswag, hellaswag_accuracy

path = download_hellaswag("data/hellaswag_val.jsonl")
result = hellaswag_accuracy(model, path, device=device, limit=300, enc=enc)

print(f"evaluated on {result['n']} examples")
print(f"  acc      : {result['acc']:.1%}")
print(f"  acc_norm : {result['acc_norm']:.1%}   (length-normalised, the quoted metric)")
print(f"  random   : 25.0%")
"""),

    md("""
## 5. Fine-tune it on Shakespeare

Two changes from pretraining:

- **Learning rate drops ~20×** (6e-4 → 3e-5). The weights are already good; a pretraining-size
  step destroys them.
- **Dropout goes to 0.1.** 1 MB of text will otherwise be memorised outright.

`--block_size 256` crops the context window, which cuts memory and speeds this up a lot for a
fine-tune where long-range context is not the point.
"""),
    code("""
!python scripts/prepare_data.py --dataset tinyshakespeare
"""),
    code("""
!python scripts/train.py \\
    --init_from gpt2 \\
    --data_dir data/tinyshakespeare \\
    --block_size 256 --batch_size 2 --grad_accum_steps 8 \\
    --learning_rate 3e-5 --warmup_steps 50 --max_steps 400 \\
    --dropout 0.1 --eval_interval 100 --eval_iters 20 \\
    --out_dir out/finetune
"""),

    md("""
## 6. Before and after

The same prompt through the base model and the fine-tuned one. The base model continues in
generic modern English; the fine-tuned one adopts the register, the speaker labels and the
line structure of a play — while keeping the grammar it learned from the web.

That is what fine-tuning does: it moves the *style*, not the *competence*.
"""),
    code("""
from gpt2.pretrained import load_checkpoint

tuned = load_checkpoint("out/finetune/ckpt_final.pt", device=device)
prompt = "ROMEO:"
idx = torch.tensor([enc.encode(prompt)], device=device)

for name, m in (("base GPT-2", model), ("fine-tuned", tuned)):
    torch.manual_seed(1337)
    out = m.generate(idx, max_new_tokens=180, temperature=0.8, top_k=50)
    print(f"{'=' * 70}\\n{name}\\n{'=' * 70}\\n{enc.decode(out[0].tolist())}\\n")
"""),

    md("""
## 7. Throughput

Useful to know what your hardware actually delivers before planning a longer run.
"""),
    code("""
!python scripts/benchmark.py --preset gpt2 --batch_size 4 --block_size 512 --steps 10
"""),

    md("""
## Where to go next

- **Multi-GPU**: `torchrun --standalone --nproc_per_node=N scripts/train.py --strategy ddp`
- **Real pretraining**: swap TinyShakespeare for FineWeb-Edu. Reproducing GPT-2 124M properly
  takes about 10B tokens and ~2 hours on 8×A100 (roughly $25-50 of rented GPU) — everything
  else in this repo is free.
- **Read the code**: `src/gpt2/model.py` is the whole architecture in about 300 lines.
"""),
])


def main() -> None:
    for name, nb in (
        ("01_train_from_scratch.ipynb", NB1),
        ("02_pretrained_and_finetune.ipynb", NB2),
    ):
        path = HERE / name
        path.write_text(json.dumps(nb, indent=1) + "\n")
        print(f"wrote {path}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
