---
title: "Llama 3.2 From Scratch - How Modern LLMs Improved on GPT-2"
description: "Week 2 of the Model Atlas: rebuild the Llama-style decoder block, compare it with GPT-2, and explain why RoPE, RMSNorm, GQA, and SwiGLU became the modern default."
date: "2026-04-11"
slug: "building-llama32-from-scratch"
tags: ["LLMs", "Transformers", "Llama", "PyTorch", "AI Engineering"]
canonical: "https://saran-io.github.io/blog/building-llama32-from-scratch/"
---

> Week 2 of my 24-week deep dive into AI models. Last week I built GPT-2 from scratch. This week I rebuilt the small Llama 3.2-style decoder block to answer one question: what actually changed between the classic GPT-2 stack and the LLM architecture most open models use today?

**Code:** [`models/02-llama32-from-scratch/code`](https://github.com/saran-io/model-atlas/tree/main/models/02-llama32-from-scratch/code)  
**Diagrams:** [`models/02-llama32-from-scratch/diagrams`](https://github.com/saran-io/model-atlas/tree/main/models/02-llama32-from-scratch/diagrams)

## TL;DR

- Llama 3.2 is still a decoder-only transformer: tokens in, next-token logits out.
- The modern upgrades are mostly engineering choices: **RMSNorm**, **RoPE**, **GQA**, and **SwiGLU**.
- Compared with GPT-2, Llama-style models are better suited for long context, efficient serving, and local deployment.
- The big lesson: modern LLMs did not replace the transformer block. They made the block cheaper, longer-context, and easier to scale.

## Why Llama After GPT-2?

GPT-2 is the cleanest starting point because it shows the original recipe: learned token embeddings, learned position embeddings, multi-head attention, GELU MLP, LayerNorm, and next-token prediction.

Llama is the best next step because it keeps the same mental model while swapping in the parts that became standard in modern open-weight LLMs. If GPT-2 teaches the skeleton, Llama teaches the production shape.

The Llama 3.2 1B model is also small enough to reason about. Meta describes the Llama 3.2 text models as optimized transformer models with **grouped-query attention** and long-context support in the public model card on [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B) and the [Llama docs](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/). Raschka's [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) is useful here because it makes these architecture differences visible instead of hiding them behind benchmark numbers.

## The Architecture

At a high level, Llama 3.2 is still:

```text
tokens -> embeddings -> repeated decoder blocks -> final norm -> LM head
```

The interesting changes are inside the decoder block.

```text
Input hidden state
  -> RMSNorm
  -> Grouped-Query Self-Attention with RoPE
  -> Residual add
  -> RMSNorm
  -> SwiGLU feed-forward network
  -> Residual add
```

![Llama 3.2 decoder block — RMSNorm, GQA + RoPE, RMSNorm, SwiGLU, residual adds](https://raw.githubusercontent.com/saran-io/model-atlas/main/models/02-llama32-from-scratch/diagrams/llama32-architecture.png)

*All diagram sources (HTML, SVG, Mermaid, Excalidraw) live in [`diagrams/`](https://github.com/saran-io/model-atlas/tree/main/models/02-llama32-from-scratch/diagrams). Editorial SVGs were authored with the [diagram-design](https://github.com/cathrynlavery/diagram-design) Claude skill; the original Excalidraw workflow that inspired the iterate-and-edit loop is described by [YF Ooi](https://yeefei.beehiiv.com/p/custom-claude-code-skill-auto-generating-updating-architecture-diagrams-with-excalidraw?utm_source=medium&utm_medium=referral&utm_campaign=excalidraw-skill).*

## What Changed From GPT-2?

| Component | GPT-2 | Llama 3.2-style stack | Why it matters |
|---|---|---|---|
| Position | Learned absolute position embeddings | RoPE | Better relative position behavior and long-context scaling |
| Attention | Multi-head attention | Grouped-query attention | Lower KV-cache memory during inference |
| Normalization | LayerNorm | RMSNorm | Simpler normalization, no mean subtraction |
| FFN | GELU MLP | SwiGLU MLP | Stronger gated feed-forward block |
| Context | 1,024 tokens in GPT-2 small | Much longer context in Llama 3.2 text models | More useful for documents, agents, and code |
| Tokenizer | GPT-2 BPE vocabulary | Larger modern tokenizer | Better coverage across code and multilingual text |

![GPT-2 vs Llama 3.2 — same skeleton, four structural upgrades](https://raw.githubusercontent.com/saran-io/model-atlas/main/models/02-llama32-from-scratch/diagrams/gpt2-vs-llama32.png)

None of these changes alter the core training objective. The model still predicts the next token. But each change removes a bottleneck that shows up when you scale model size, context length, or serving traffic.

## The Four Modern Upgrades

### 1. RMSNorm

GPT-2 uses LayerNorm:

```python
y = (x - mean(x)) / sqrt(var(x) + eps)
```

Llama uses RMSNorm:

```python
y = x / sqrt(mean(x * x) + eps)
```

RMSNorm skips mean subtraction. That sounds small, but at LLM scale "small" operations happen billions of times. The idea is simple: normalize the magnitude of the vector, then learn a scale.

![LayerNorm vs RMSNorm — same regularizing effect, two operations removed](https://raw.githubusercontent.com/saran-io/model-atlas/main/models/02-llama32-from-scratch/diagrams/rmsnorm-vs-layernorm.png)

### 2. RoPE

GPT-2 learns a position embedding table. Position 0 has a vector. Position 1 has a vector. The model adds those vectors to token embeddings.

RoPE does something different: it rotates query and key vectors based on position before computing attention. This lets attention scores carry relative position information directly.

The useful intuition:

```text
GPT-2: "add a learned position vector to the token"
RoPE:  "rotate attention vectors so distance is built into Q dot K"
```

That is one reason Llama-style models are a better foundation for long-context systems.

![RoPE rotates Q and K by position so the dot product carries relative distance](https://raw.githubusercontent.com/saran-io/model-atlas/main/models/02-llama32-from-scratch/diagrams/rope-rotation.png)

### 3. Grouped-Query Attention

In classic multi-head attention, every query head has its own key and value head. During generation, the server stores all previous keys and values in the KV cache. Long context makes this expensive.

Grouped-query attention keeps many query heads but shares fewer key/value heads across them.

```text
MHA: 32 query heads, 32 key heads, 32 value heads
GQA: 32 query heads, 8 key heads, 8 value heads
```

The output still has many query heads, but the cache is smaller. That matters in production because KV-cache memory is often the reason long-context inference gets slow or expensive.

![Multi-Head vs Grouped-Query Attention — fewer K/V heads serve more Q heads](https://raw.githubusercontent.com/saran-io/model-atlas/main/models/02-llama32-from-scratch/diagrams/attention-evolution.png)

![KV-cache memory at long context — GQA cuts the bill 4×](https://raw.githubusercontent.com/saran-io/model-atlas/main/models/02-llama32-from-scratch/diagrams/kv-cache-memory.png)

### 4. SwiGLU

GPT-2 uses a simple GELU feed-forward network:

```text
x -> Linear -> GELU -> Linear
```

Llama uses a gated feed-forward block:

```text
x -> SiLU(gate_proj(x)) * up_proj(x) -> down_proj
```

One branch decides what to let through; the other branch carries candidate features. The multiplication makes the feed-forward layer more expressive without changing the basic transformer loop.

![SwiGLU feed-forward — gate branch (SiLU) ⊙ candidate branch, then down-projection](https://raw.githubusercontent.com/saran-io/model-atlas/main/models/02-llama32-from-scratch/diagrams/swiglu-gate-flow.png)

## What I Built

The code for this week implements the core Llama-style decoder in PyTorch:

- token embeddings
- RMSNorm
- RoPE
- grouped-query self-attention
- SwiGLU feed-forward network
- tied LM head
- small config that runs on a laptop

Run it locally:

```bash
cd models/02-llama32-from-scratch/code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python llama32.py --prompt "The future of open models is" --max_new_tokens 40
```

The default config is intentionally tiny. It is not trying to reproduce Meta's weights. It is a readable implementation of the architecture pattern, so the math lines up with the diagrams.

![Llama 3.2 end-to-end — tokens in, next-token logits out, all four upgrades inside the decoder block](https://raw.githubusercontent.com/saran-io/model-atlas/main/models/02-llama32-from-scratch/diagrams/inference-pipeline.png)

## Where This Applies

Llama-style models are useful when you care about:

1. **Local AI features** - small open-weight models can run on laptops, edge devices, or private servers.
2. **Private enterprise workflows** - teams can host models closer to sensitive data.
3. **Agent backends** - long context plus efficient KV cache helps tool-using systems carry more state.
4. **RAG pipelines** - open models can be tuned, quantized, routed, and evaluated inside your own stack.
5. **Model learning** - Llama is the architecture family you need to understand before reading most modern open LLM repos.

## Production Lesson

At Tekvo AI, most model decisions are not "which benchmark is highest?" They are tradeoffs:

- How much context does the task really need?
- What is the latency budget?
- Can we send this data to a hosted API?
- Does the model need tool use, coding ability, or multilingual coverage?
- Can a small local model handle 80% of the traffic?

Llama matters because it gives builders a model family they can inspect, deploy, quantize, and adapt. You do not get that with a closed API.

## Remember

- GPT-2 and Llama are the same species: decoder-only transformers trained for next-token prediction.
- RMSNorm simplifies normalization.
- RoPE moves position information into attention instead of adding a learned position vector.
- GQA reduces KV-cache pressure, which matters for long-context inference.
- SwiGLU makes the feed-forward block stronger through gating.

## Verdict

GPT-2 is the best model to learn the transformer. Llama is the best next model to learn what modern LLM engineering looks like.

If Week 1 answered "how does a language model work?", Week 2 answers "what did modern open models change to make this practical at scale?"

---

*Week 2/24. Next up: Mixture-of-Experts with Mistral/Mixtral, where the model gets bigger without activating every parameter on every token.*
