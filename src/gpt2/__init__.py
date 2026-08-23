"""GPT-2 (124M) implemented from scratch in PyTorch.

    from gpt2 import GPT, GPTConfig
    model = GPT(GPTConfig())              # 124,439,808 parameters
    model = GPT.from_pretrained("gpt2")   # OpenAI's weights, our implementation
"""

from .config import GPTConfig, TrainConfig
from .model import GPT, MLP, Block, CausalSelfAttention, LayerNorm
from .tokenizer import BPETokenizer, CharTokenizer, get_tokenizer

__version__ = "0.1.0"

__all__ = [
    "GPT",
    "GPTConfig",
    "TrainConfig",
    "Block",
    "CausalSelfAttention",
    "MLP",
    "LayerNorm",
    "BPETokenizer",
    "CharTokenizer",
    "get_tokenizer",
]
