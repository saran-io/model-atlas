# I Built GPT-2 From Scratch — Here's What Every AI Engineer Should Know

> Week 1 of my 24-week deep dive into AI models. I'm Sayora, co-founder of Tekvo AI, where we build agentic systems. This week I implemented GPT-2 from scratch in PyTorch — every layer, every matrix multiply. Here's everything I learned.

## TL;DR
- GPT-2 is the architecture behind every major LLM today — GPT-4, Claude, Llama, DeepSeek all descend from it
- I implemented all 117M parameters from scratch and loaded the real weights to prove correctness
- The core insight: self-attention is just a learned lookup table, and everything else is surprisingly simple

## Why GPT-2 in 2026?

You might think GPT-2 is ancient history. It's not. Here's why:

Every model I'll cover in the next 23 weeks — GPT-4o, Claude Opus, Llama 4, DeepSeek R1 — uses the same fundamental building blocks that GPT-2 introduced to the mainstream. The differences are in scale, training data, and clever optimizations. But the bones are the same.

If you understand GPT-2 deeply, you understand 80% of every modern LLM.

## The Architecture

GPT-2 is a **decoder-only transformer**. Let's break down what that means.

The original Transformer (Vaswani et al., 2017) had an encoder and a decoder. GPT-2 threw away the encoder and kept only the decoder. Why? Because for text generation, you only need to predict the next token given previous tokens. The encoder was designed for understanding input sequences (useful for translation), but for pure generation, it's unnecessary complexity.

![GPT-2 Architecture](../diagrams/gpt2-architecture.png)

### The Components (In Order)

**1. Tokenizer (BPE)**
Before anything touches the neural network, text gets split into tokens using Byte Pair Encoding. "The cat sat" becomes `[464, 3797, 3332]`. BPE is a compression algorithm turned tokenizer — it merges frequent character pairs into single tokens. This is why "tokenization" is one token but " tokenization" (with a space) is two.

**2. Token + Position Embeddings**
Each token ID maps to a 768-dimensional vector (learned). Each position (0 to 1023) also maps to a 768-dimensional vector (also learned — NOT sinusoidal like the original Transformer). These are added together. That's it. The model has to figure out word meaning AND position from this sum.

**3. 12 Transformer Blocks** (this is where the magic happens)

Each block has two sub-layers:

**Self-Attention:** "For each token, which other tokens should I look at?"
```python
# The entire self-attention mechanism in ~10 lines
Q, K, V = input @ W_Q, input @ W_K, input @ W_V
scores = (Q @ K.T) / sqrt(d_k)          # How similar is each query to each key?
scores = scores.masked_fill(causal, -inf) # Can't look at future tokens
weights = softmax(scores)                 # Normalize to probabilities
output = weights @ V                      # Weighted sum of values
```

**Feed-Forward Network:** "Now that I know what to look at, what do I do with it?"
```python
# Expand, activate, contract
output = linear2(gelu(linear1(input)))    # 768 -> 3072 -> 768
```

Both use **residual connections** (add the input back to the output) and **pre-norm** (LayerNorm before, not after — GPT-2's key divergence from the original Transformer).

**4. LM Head**
The final layer projects from 768 dimensions back to 50,257 (vocabulary size). The output is a probability distribution over every possible next token. GPT-2 ties these weights with the token embeddings — the same matrix that converts tokens to vectors is used (transposed) to convert vectors back to token probabilities.

## What I Built

I implemented every component from scratch in PyTorch — no HuggingFace wrappers, no shortcuts. Then I loaded the real GPT-2 weights into my implementation and verified it produces identical outputs.

```python
# Load real weights into our from-scratch model
model = load_pretrained_gpt2()

# Generate text — this uses OUR implementation, not HuggingFace
output = generate(model, "The future of AI is", max_tokens=50)
```

### Attention Visualization

I also built a visualizer to see what each attention head learns:

```
Attention Pattern — Layer 0, Head 0
        The     cat     sat      on
 The   1.000   0.000   0.000   0.000
 cat   0.412   0.588   0.000   0.000
 sat   0.183   0.304   0.513   0.000
  on   0.091   0.156   0.287   0.466
```

Notice: each row sums to 1.0 (it's a probability distribution), and the upper triangle is all zeros (causal mask — can't see the future).

Different heads learn different patterns. Some are "positional" (attend to the previous token). Some are "semantic" (attend to syntactically related tokens). This emerges purely from training — nobody programs these patterns.

## Surprises & Gotchas

### 1. The scaling factor matters more than you think
Dividing attention scores by `√d_k` seems like a minor detail. It's not. Without it, as dimensions grow, dot products become huge, softmax saturates to one-hot vectors, and gradients vanish. I tested without scaling — the model produces garbage even with correct weights.

### 2. Weight tying is elegant
The token embedding matrix (50,257 × 768) is the *same* matrix used in the output head, just transposed. This means "the vector for token X" and "predicting token X" share the same learned representation. It saves parameters AND works better.

### 3. Pre-norm vs post-norm is a big deal
GPT-2 applies LayerNorm *before* attention/FFN. The original Transformer does it *after*. This seems minor but it dramatically improves training stability at scale. Every modern LLM uses pre-norm now.

### 4. GELU, not ReLU
GPT-2 uses GELU activation (a smooth approximation of ReLU that allows small negative values through). This was an unusual choice in 2019 but is now standard. The intuition: ReLU kills information by zeroing negatives entirely; GELU preserves some of it.

## System Design: Where GPT-2 Architecture Fits

![System Design](../diagrams/self-attention.png)

While you won't deploy GPT-2 itself in production, understanding its architecture tells you exactly what's happening inside every API call you make:

### Why This Matters for Agentic Systems (Tekvo AI Perspective)
At Tekvo AI, every agent we build calls an LLM that's a direct descendant of GPT-2. Understanding the architecture helps us:
- **Debug failures:** When an agent hallucinates, it's because the attention mechanism assigned high weight to irrelevant context
- **Optimize costs:** Token count directly maps to compute because of the T² attention complexity
- **Choose models:** Dense transformers (GPT-2's architecture) vs MoE (Mixtral's) have fundamentally different cost profiles

## What Changed Since GPT-2

| Feature | GPT-2 (2019) | Modern LLMs (2025-26) |
|---------|--------------|----------------------|
| Positional encoding | Learned, absolute | RoPE (relative, extrapolates) |
| Attention | Full O(n²) | GQA, MQA, sliding window |
| Activation | GELU | SwiGLU |
| Norm | Pre-norm LayerNorm | RMSNorm |
| Architecture | Dense | Dense or MoE |
| Context | 1,024 tokens | 128K–1M+ tokens |
| Parameters | 117M–1.5B | 8B–1.8T |

These are optimizations, not reinventions. The core idea — stack self-attention and FFN blocks, train to predict the next token — hasn't changed.

## Verdict

GPT-2 isn't a model you'd deploy. It's a model you must **understand**. Every architectural choice here — decoder-only, autoregressive, pre-norm, weight tying — became the template for the $100B+ models running today.

If you can implement GPT-2 from scratch, you can read any modern LLM's code and understand what's happening.

---

*This is week 1 of my 24-week AI models deep dive. Every week I pick one model, build something real with it, and share what I learn. Follow along:*

*Code: [GitHub link]*

*Next week: GPT-4o vs GPT-4.1 — I'm running 500 agentic tasks on both. Stay tuned.*
