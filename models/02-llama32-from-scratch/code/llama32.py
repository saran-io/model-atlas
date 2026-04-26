"""Tiny Llama 3.2-style decoder for architecture learning.

This is intentionally small and randomly initialized. It demonstrates the
modern Llama block shape: RMSNorm, RoPE, grouped-query attention, and SwiGLU.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LlamaConfig:
    vocab_size: int = 256
    max_seq_len: int = 256
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 2
    intermediate_size: int = 384
    rms_norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


class ByteTokenizer:
    """Minimal byte tokenizer so the demo has no external tokenizer dependency."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, ids: list[int]) -> str:
        return bytes([idx % 256 for idx in ids]).decode("utf-8", errors="replace")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * rms


def precompute_rope_frequencies(head_dim: int, seq_len: int, theta: float = 500_000.0) -> torch.Tensor:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(seq_len).float()
    freqs = torch.outer(positions, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs_complex: torch.Tensor) -> torch.Tensor:
    batch, seq_len, n_heads, head_dim = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(batch, seq_len, n_heads, head_dim // 2, 2))
    freqs_complex = freqs_complex[:seq_len].unsqueeze(0).unsqueeze(2)
    rotated = x_complex * freqs_complex
    return torch.view_as_real(rotated).flatten(3).type_as(x)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        if config.n_heads % config.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = config.n_heads // config.n_kv_heads

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, freqs_complex)
        k = apply_rope(k, freqs_complex)

        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.head_dim)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, config.rms_norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.dim, config.rms_norm_eps)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_complex)
        return x + self.ffn(self.ffn_norm(x))


class TinyLlama(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.rms_norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

        freqs = precompute_rope_frequencies(config.head_dim, config.max_seq_len)
        self.register_buffer("freqs_complex", freqs, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.size(1) > self.config.max_seq_len:
            tokens = tokens[:, -self.config.max_seq_len :]

        x = self.tok_embeddings(tokens)
        freqs_complex = self.freqs_complex.to(x.device)
        for layer in self.layers:
            x = layer(x, freqs_complex)
        return self.output(self.norm(x))

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int, temperature: float, top_k: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx = tokens[:, -self.config.max_seq_len :]
            logits = self(idx)[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny Llama 3.2-style decoder.")
    parser.add_argument("--prompt", default="The future of open models is")
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show_config", action="store_true")
    args = parser.parse_args()

    config = LlamaConfig()
    if args.show_config:
        print(asdict(config))
        return

    torch.manual_seed(args.seed)
    tokenizer = ByteTokenizer()
    model = TinyLlama(config).eval()

    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    output_ids = model.generate(input_ids, args.max_new_tokens, args.temperature, args.top_k)[0].tolist()
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
