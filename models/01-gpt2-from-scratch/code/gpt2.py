"""
GPT-2 (117M) — Built From Scratch
Week 1 of the AI Models Deep Dive by Saran (Tekvo)

This is NOT a wrapper around HuggingFace. Every component is implemented
from first principles so you understand exactly what happens at each layer.

Usage:
    python gpt2.py --prompt "The future of AI is"
    python gpt2.py --prompt "The future of AI is" --max_tokens 100
    python gpt2.py --visualize  # Show attention patterns
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import argparse


# =============================================================================
# Configuration — matches the original GPT-2 117M (small)
# =============================================================================

@dataclass
class GPT2Config:
    vocab_size: int = 50257      # BPE vocabulary size
    context_length: int = 1024   # Maximum sequence length
    n_layers: int = 12           # Number of transformer blocks
    n_heads: int = 12            # Number of attention heads
    d_model: int = 768           # Embedding dimension
    d_ff: int = 3072             # Feed-forward inner dimension (4 * d_model)
    dropout: float = 0.1
    bias: bool = True


# =============================================================================
# Core Components — The Building Blocks
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    The heart of the transformer.

    Self-attention answers: "For each token, which other tokens should I pay
    attention to, and how much?"

    Multi-head = running multiple attention patterns in parallel, each
    learning different relationships (syntax, semantics, position, etc.)
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        # Q, K, V projections — combined into one linear for efficiency
        # This is a single [d_model, 3 * d_model] matrix that we split after
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)

        # Output projection — combines all heads back together
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask — prevents attending to future tokens
        # This is what makes it "autoregressive" (left-to-right generation)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length))
                 .view(1, 1, config.context_length, config.context_length)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape  # batch, sequence_length, d_model

        # Project input into Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape for multi-head: (B, T, d_model) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # Why scale by 1/sqrt(d_k)? Without it, dot products grow large with
        # dimension, pushing softmax into saturated regions with tiny gradients.
        attn_weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Apply causal mask — set future positions to -inf so softmax gives 0
        attn_weights = attn_weights.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float('-inf')
        )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        out = attn_weights @ v  # (B, n_heads, T, head_dim)

        # Concatenate heads: (B, n_heads, T, head_dim) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final projection
        out = self.resid_dropout(self.out_proj(out))

        return out, attn_weights


class FeedForward(nn.Module):
    """
    The "thinking" layer — a simple expand-then-contract MLP.

    Attention decides WHAT to look at.
    FFN decides WHAT TO DO with that information.

    Architecture: d_model -> 4*d_model (GELU) -> d_model
    The 4x expansion is a design choice from the original Transformer paper.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GELU (not ReLU) — GPT-2's activation function
        # GELU is smoother than ReLU, allowing small negative values through
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    One block of the GPT-2 transformer.

    Pre-norm architecture (GPT-2's key difference from the original Transformer):
        x -> LayerNorm -> Attention -> + residual
        x -> LayerNorm -> FFN -> + residual

    The original Transformer used post-norm (apply LN after attention/FFN).
    Pre-norm trains more stably — this became the standard for all modern LLMs.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm + residual connection
        attn_out, attn_weights = self.attn(self.ln1(x))
        x = x + attn_out                   # Residual connection 1
        x = x + self.ffn(self.ln2(x))      # Residual connection 2
        return x, attn_weights


# =============================================================================
# The Full GPT-2 Model
# =============================================================================

class GPT2(nn.Module):
    """
    GPT-2: A decoder-only transformer language model.

    What GPT-2 dropped from the original Transformer:
    - No encoder (decoder-only)
    - No cross-attention (only self-attention)
    - Pre-norm instead of post-norm
    - Learned positional embeddings instead of sinusoidal

    The simplicity is the point — this architecture scaled to GPT-4.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        # Token embeddings: vocab_size -> d_model
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Position embeddings: context_length -> d_model (LEARNED, not sinusoidal)
        self.pos_emb = nn.Embedding(config.context_length, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        # The transformer blocks — this is where the magic happens
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final layer norm (applied before the output head)
        self.ln_f = nn.LayerNorm(config.d_model)

        # Output head: d_model -> vocab_size (predicts next token)
        # NOTE: GPT-2 ties this with token_emb weights (weight tying)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Weight tying

    def forward(
        self,
        input_ids: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
        B, T = input_ids.shape

        # Create position indices: [0, 1, 2, ..., T-1]
        positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device)

        # Embed tokens + positions
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)

        # Pass through all transformer blocks
        all_attn_weights = []
        for block in self.blocks:
            x, attn_weights = block(x)
            if return_attention:
                all_attn_weights.append(attn_weights)

        # Final norm + project to vocabulary
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits, all_attn_weights if return_attention else None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Loading Real GPT-2 Weights from HuggingFace
# =============================================================================

def load_pretrained_gpt2(model_size: str = "gpt2") -> GPT2:
    """
    Load real GPT-2 weights into our from-scratch implementation.

    This proves our implementation is correct — if the architecture doesn't
    match exactly, the weights won't load and outputs will be garbage.
    """
    from transformers import GPT2LMHeadModel

    print(f"Loading pretrained {model_size} weights...")
    hf_model = GPT2LMHeadModel.from_pretrained(model_size)
    hf_sd = hf_model.state_dict()

    config = GPT2Config()
    model = GPT2(config)

    # Map HuggingFace weight names to our weight names
    mapping = {}
    for i in range(config.n_layers):
        hf_prefix = f"transformer.h.{i}"
        our_prefix = f"blocks.{i}"

        # Attention QKV (HuggingFace combines them as c_attn)
        mapping[f"{hf_prefix}.attn.c_attn.weight"] = f"{our_prefix}.attn.qkv_proj.weight"
        mapping[f"{hf_prefix}.attn.c_attn.bias"] = f"{our_prefix}.attn.qkv_proj.bias"
        mapping[f"{hf_prefix}.attn.c_proj.weight"] = f"{our_prefix}.attn.out_proj.weight"
        mapping[f"{hf_prefix}.attn.c_proj.bias"] = f"{our_prefix}.attn.out_proj.bias"

        # FFN
        mapping[f"{hf_prefix}.mlp.c_fc.weight"] = f"{our_prefix}.ffn.fc1.weight"
        mapping[f"{hf_prefix}.mlp.c_fc.bias"] = f"{our_prefix}.ffn.fc1.bias"
        mapping[f"{hf_prefix}.mlp.c_proj.weight"] = f"{our_prefix}.ffn.fc2.weight"
        mapping[f"{hf_prefix}.mlp.c_proj.bias"] = f"{our_prefix}.ffn.fc2.bias"

        # Layer norms
        mapping[f"{hf_prefix}.ln_1.weight"] = f"{our_prefix}.ln1.weight"
        mapping[f"{hf_prefix}.ln_1.bias"] = f"{our_prefix}.ln1.bias"
        mapping[f"{hf_prefix}.ln_2.weight"] = f"{our_prefix}.ln2.weight"
        mapping[f"{hf_prefix}.ln_2.bias"] = f"{our_prefix}.ln2.bias"

    # Global weights
    mapping["transformer.wte.weight"] = "token_emb.weight"
    mapping["transformer.wpe.weight"] = "pos_emb.weight"
    mapping["transformer.ln_f.weight"] = "ln_f.weight"
    mapping["transformer.ln_f.bias"] = "ln_f.bias"

    # Load weights (HuggingFace uses Conv1D which stores weights transposed)
    our_sd = model.state_dict()
    for hf_name, our_name in mapping.items():
        if hf_name not in hf_sd:
            print(f"  Warning: {hf_name} not found in HF model")
            continue

        w = hf_sd[hf_name]

        # HuggingFace GPT-2 uses Conv1D (transposed weights) for attention & FFN
        if "attn.c_attn" in hf_name or "attn.c_proj" in hf_name or "mlp" in hf_name:
            if w.dim() == 2:
                w = w.t()

        our_sd[our_name] = w

    model.load_state_dict(our_sd, strict=False)
    model.eval()

    print(f"Loaded {model.count_parameters():,} parameters")
    return model


# =============================================================================
# Text Generation
# =============================================================================

@torch.no_grad()
def generate(
    model: GPT2,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    """
    Autoregressive text generation.

    The model predicts one token at a time, appends it to the input,
    and repeats. This is how ALL autoregressive LLMs generate text.
    """
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    for _ in range(max_tokens):
        # Crop to context window
        input_cropped = input_ids[:, -model.config.context_length:]

        # Forward pass
        logits, _ = model(input_cropped)

        # Get logits for the last token only
        logits = logits[:, -1, :] / temperature

        # Top-k sampling: zero out everything except top-k tokens
        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, [-1]]] = float('-inf')

        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append and continue
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0].tolist())


# =============================================================================
# Attention Visualization
# =============================================================================

def visualize_attention(model: GPT2, text: str, layer: int = 0, head: int = 0):
    """
    Visualize what the attention heads are looking at.

    Different heads learn different patterns:
    - Some attend to the previous token (local/positional)
    - Some attend to specific syntactic roles (subject, verb)
    - Some attend broadly (semantic/global)
    """
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_ids = torch.tensor([tokenizer.encode(text)])
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    with torch.no_grad():
        _, all_attn = model(input_ids, return_attention=True)

    attn = all_attn[layer][0, head].cpu().numpy()  # (T, T)

    print(f"\nAttention Pattern — Layer {layer}, Head {head}")
    print(f"Tokens: {tokens}\n")

    # Print attention matrix as a simple heatmap
    header = "".join(f"{t[:6]:>7}" for t in tokens)
    print(f"{'':>7}{header}")

    for i, token in enumerate(tokens):
        row = "".join(f"{attn[i][j]:7.3f}" for j in range(len(tokens)))
        print(f"{token[:6]:>7}{row}")

    print("\nHigher values = more attention. Each row sums to 1.0.")
    print("Notice the causal mask: tokens can only attend to previous tokens.")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-2 From Scratch")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is")
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--visualize", action="store_true", help="Visualize attention patterns")
    parser.add_argument("--viz_text", type=str, default="The cat sat on the mat")
    parser.add_argument("--viz_layer", type=int, default=0)
    parser.add_argument("--viz_head", type=int, default=0)
    args = parser.parse_args()

    print("=" * 60)
    print("GPT-2 From Scratch — Week 1 of AI Models Deep Dive")
    print("by Saran (Tekvo)")
    print("=" * 60)

    # Load the real GPT-2 weights into our implementation
    model = load_pretrained_gpt2()

    if args.visualize:
        visualize_attention(model, args.viz_text, args.viz_layer, args.viz_head)
    else:
        print(f"\nPrompt: {args.prompt}")
        print(f"Generating {args.max_tokens} tokens (temp={args.temperature}, top_k={args.top_k})...")
        print("-" * 60)
        output = generate(model, args.prompt, args.max_tokens, args.temperature, args.top_k)
        print(output)
        print("-" * 60)
