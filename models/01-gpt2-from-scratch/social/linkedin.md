# Week 01 — LinkedIn (GPT-2 from scratch)

Copy-paste ready. Replace `[BLOG_URL]` after you publish (e.g. `https://sayora.dev/blog/gpt2-from-scratch`).

---

I just shipped **Week 1** of my 24-week “models from the inside out” series.

This week: **GPT-2, implemented from scratch in PyTorch** — not a HuggingFace wrapper, every matmul explicit — then **real OpenAI weights loaded** to prove the implementation matches the reference.

Why GPT-2 in 2026?

Because GPT-4 class models, Claude, Llama, and the rest are still **the same architectural story**: decoder-only transformer, autoregressive training, multi-head self-attention, pre-norm residuals. Scale and data changed; the skeleton didn’t.

Three things that stuck with me:

**1 — Self-attention is a learned lookup table.**  
Q, K, V learn who should attend to whom. Nobody hand-wires “grammar head” vs “semantics head”; patterns emerge from training.

**2 — The 1/√d scaling is load-bearing.**  
I removed it as a sanity check. Same weights, **garbage outputs**. Tiny detail, huge effect on softmax and gradients.

**3 — Weight tying is elegant.**  
Token embeddings and the LM head share weights (transposed). Fewer parameters, and it works better in practice.

I also added a small **animated explainer** (pure HTML/CSS) so you can *see* token flow, the repeated blocks, and a toy causal mask — useful when you’re teaching or onboarding engineers.

**Open source (code + diagrams + post draft):**  
https://github.com/saran-io/model-atlas

**Blog post:**  
[BLOG_URL]

At Tekvo AI we build agentic systems every day. Understanding model internals isn’t academic — it’s how we debug failures, reason about cost/latency vs context length, and pick the right model for each agent step.

**Next week:** GPT-4o vs GPT-4.1 on real agentic workloads.

If you’re learning LLMs, start from the repo’s `models/01-gpt2-from-scratch` — run `gpt2.py`, then read `DEEP-DIVE.md`.

#AIEngineering #LLMs #PyTorch #Transformers #GPT2 #MachineLearning #AgenticAI
