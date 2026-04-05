---
title: "I Built GPT-2 From Scratch — Here's What Every AI Engineer Should Know"
description: "Week 1 of a 24-week models series: a from-scratch PyTorch GPT-2, real weights, and the lessons that carry straight to GPT-4 and Claude."
date: "2026-04-04"
slug: "gpt2-from-scratch"
tags: ["LLMs", "Transformers", "PyTorch", "GPT-2", "AI Engineering"]
canonical: "https://saran-io.github.io/blog/building-gpt2-from-scratch/"
---

> Week 1 of my 24-week deep dive into AI models. I'm **Saran**, co-founder of [Tekvo](https://tekvo.ai), where we build agentic systems. This week I implemented GPT-2 from scratch in PyTorch — every layer, every matrix multiply. Here's everything I learned.

**Interactive:** Open [`models/01-gpt2-from-scratch/explainer/index.html`](https://github.com/saran-io/model-atlas/tree/main/models/01-gpt2-from-scratch/explainer) in a browser (clone or download the repo), or copy the `explainer/` folder into your site’s static assets — e.g. `https://saran-io.github.io/labs/gpt2-explainer/` — so readers get a lightweight animated walkthrough of token flow, the block stack, and causal attention.

## TL;DR

- GPT-2 is the architectural ancestor of today's frontier LLMs — same bones, different scale.
- I implemented the full stack in PyTorch and loaded real OpenAI weights to verify correctness.
- Self-attention is a learned lookup table; the \(1/\sqrt{d_k}\) scaling is non-negotiable in practice.

## Why GPT-2 in 2026?

Every model in this series — GPT-4 class, Claude, Llama, DeepSeek — stacks the same primitives GPT-2 popularized: decoder-only transformer, autoregressive training, multi-head self-attention, pre-norm residuals. The differences are scale, data, and engineering. If you grok GPT-2, you can read almost any LLM codebase.

## The architecture

GPT-2 is a **decoder-only transformer**. It dropped the encoder from the original Transformer because next-token prediction only needs a left-to-right stack.

![GPT-2 architecture (data flow)](https://raw.githubusercontent.com/saran-io/model-atlas/main/models/01-gpt2-from-scratch/diagrams/gpt2-architecture.png)

### Components (in order)

**1. Tokenizer (BPE)**  
Text becomes token IDs. BPE merges frequent fragments — why `"tokenization"` and `" tokenization"` tokenize differently.

**2. Token + position embeddings**  
Each token and each position maps to a learned 768-d vector; they are summed. GPT-2 uses learned absolute positions (not sinusoidal).

**3. Twelve transformer blocks**  
Each block: LayerNorm → multi-head self-attention → residual → LayerNorm → FFN (768 → 3072 → 768, GELU) → residual. Pre-norm matters for training stability.

**4. LM head**  
Projects to vocabulary logits; in GPT-2, weights are **tied** with token embeddings (same matrix, transposed).

### Self-attention in one glance

![Self-attention steps](https://raw.githubusercontent.com/saran-io/model-atlas/main/models/01-gpt2-from-scratch/diagrams/self-attention.png)

```python
Q, K, V = x @ W_Q, x @ W_K, x @ W_V
scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
scores = scores.masked_fill(causal_mask, float("-inf"))
weights = softmax(scores, dim=-1)
out = weights @ V
```

## What I built

Runnable code lives in the repo — no HuggingFace `AutoModel` wrapper for the core forward pass: it's explicit modules you can line up with a diagram.

```bash
cd models/01-gpt2-from-scratch/code
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python gpt2.py --prompt "The future of AI is" --max_tokens 80
python gpt2.py --visualize --viz_text "The cat sat on the mat"
```

**Repo:** [github.com/saran-io/model-atlas](https://github.com/saran-io/model-atlas)  
**Deep dive (longer version):** [models/01-gpt2-from-scratch/DEEP-DIVE.md](https://github.com/saran-io/model-atlas/blob/main/models/01-gpt2-from-scratch/DEEP-DIVE.md)

## Surprises and gotchas

1. **Scaling by \(1/\sqrt{d_k}\)** — Without it, dot products blow up, softmax saturates, gradients starve. Same weights, wrong behavior.
2. **Weight tying** — Input embeddings and the output projection share weights: fewer parameters, better results in practice.
3. **Pre-norm vs post-norm** — GPT-2 normalizes *before* each sublayer; the original paper often showed post-norm. Pre-norm is what scaled.
4. **GELU** — Smooth activations beat hard ReLU for these depths; modern stacks often use SwiGLU, but the pattern is the same: nonlinearity in the FFN bottleneck.

## Why this matters for agentic systems

At **Tekvo** we live inside LLM APIs all day. Internals are not academic: they explain why context length hits cost and latency, why hallucinations correlate with attention over irrelevant spans, and why tokenizer quirks become production bugs.

## What changed since GPT-2

| | GPT-2 (2019) | Modern LLMs |
|---|--------------|-------------|
| Position | Learned absolute | RoPE, ALiBi, … |
| Attention | Full, dense | GQA/MQA, sliding windows, … |
| FFN | GELU MLP | Often SwiGLU |
| Norm | LayerNorm | Often RMSNorm |
| Context | 1,024 tokens | 128K–1M+ |

Same story: predict the next token by mixing attention and FFN blocks.

## Verdict

You do not ship GPT-2 to production in 2026. You **study** it — so every stack trace, paper, and API doc has a clear physical picture behind it.

---

*Week 1/24. Next up: GPT-4o vs GPT-4.1 on real agentic tasks.*  
*Star the repo if you want the series in your feed:* [github.com/saran-io/model-atlas](https://github.com/saran-io/model-atlas)
