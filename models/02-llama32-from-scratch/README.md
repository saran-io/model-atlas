# 02 - Llama 3.2 From Scratch

> **Difficulty:** 3/5 | **Time to run:** 5 min | **GPU required:** No

Week 2 compares the classic GPT-2 block with a modern Llama-style decoder block. The goal is not to clone Meta's checkpoint; it is to make the architectural upgrades readable in code: RMSNorm, RoPE, grouped-query attention, and SwiGLU.

## Key Takeaways

1. **Llama is still a decoder-only transformer** - the core loop is attention + feed-forward + residuals.
2. **RoPE replaces learned absolute positions** - position is injected into Q/K rotations instead of a learned table.
3. **GQA reduces KV-cache memory** - many query heads share fewer key/value heads during inference.
4. **RMSNorm and SwiGLU are modern defaults** - simpler normalization and a stronger gated FFN.

## Architecture

```text
Input tokens
  -> Token embeddings
  -> N x [RMSNorm -> GQA with RoPE -> residual -> RMSNorm -> SwiGLU FFN -> residual]
  -> Final RMSNorm
  -> LM head
  -> Next-token logits
```

Architecture source: [diagrams/llama32-architecture.mmd](diagrams/llama32-architecture.mmd)

## Quick Start

```bash
cd code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python llama32.py --prompt "The future of open models is" --max_new_tokens 40
python llama32.py --show_config
```

The demo uses a tiny randomly initialized config by default so it runs quickly on CPU. It teaches tensor shapes and model flow; it is not a pretrained Llama checkpoint.

## Diagrams

| File | Purpose |
|---|---|
| `diagrams/llama32-architecture.mmd` | Main architecture flow |
| `diagrams/gpt2-vs-llama32.mmd` | GPT-2 vs Llama comparison |
| `diagrams/attention-evolution.mmd` | MHA vs GQA intuition |
| `diagrams/llama32-architecture.excalidraw` | Editable Excalidraw source |

## Experiments to Try

| Experiment | Command | What You'll Learn |
|---|---|---|
| Inspect model config | `python llama32.py --show_config` | How heads, KV heads, layers, and dimensions relate |
| Longer generation | `python llama32.py --max_new_tokens 100` | Random weights drift quickly without training |
| Change GQA ratio | Edit `num_key_value_heads` | How fewer KV heads change the cache shape |
| Disable RoPE mentally | Compare with GPT-2 code | Where position information enters the model |

## What Changed Since GPT-2

| Design Choice | GPT-2 | Llama 3.2-style |
|---|---|---|
| Position | Learned absolute embeddings | RoPE |
| Attention | Multi-head attention | Grouped-query attention |
| Norm | LayerNorm | RMSNorm |
| FFN | GELU MLP | SwiGLU |
| Context | Short fixed context | Long-context oriented |

---

**Blog (publish-ready):** [blog/week-02-llama32-from-scratch.md](../../blog/week-02-llama32-from-scratch.md)  
**Previous week:** [GPT-2 From Scratch](../01-gpt2-from-scratch/README.md)
