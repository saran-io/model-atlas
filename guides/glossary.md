# Glossary

Plain-english explanations of AI model terminology. Referenced throughout this repo.

---

**Attention / Self-Attention** — The mechanism that lets each token "look at" other tokens to understand context. See [01-gpt2-from-scratch](../models/01-gpt2-from-scratch) for implementation.

**Autoregressive** — Generating one token at a time, left-to-right. Each new token is conditioned on all previous tokens.

**BPE (Byte Pair Encoding)** — A tokenization algorithm that builds vocabulary by repeatedly merging the most frequent pair of characters/subwords.

**Causal Mask** — A triangular mask that prevents tokens from attending to future positions. This is what makes generation work left-to-right.

**Context Window** — The maximum number of tokens a model can process at once. GPT-2: 1K, GPT-4o: 128K, Gemini: 1M+.

**Decoder-Only** — A transformer that only has the "decoder" part (with causal masking). All modern LLMs (GPT, Claude, Llama) use this. The original Transformer had both encoder and decoder.

**Dense Model** — A model where every parameter is active for every input. Contrast with MoE.

**Embedding** — A fixed-size vector representation of a token, sentence, or document. Used for search, RAG, and similarity.

**FFN (Feed-Forward Network)** — The MLP layer in each transformer block. Attention decides *what to look at*; FFN decides *what to do with it*.

**Fine-tuning** — Continuing training on a specific dataset to adapt a model. LoRA/QLoRA are parameter-efficient methods.

**GQA (Grouped Query Attention)** — Sharing K,V heads across multiple Q heads. Reduces memory during inference. Used by Llama 3/4.

**GGUF** — A file format for quantized models, used by llama.cpp for CPU inference.

**Hallucination** — When a model generates confident-sounding but factually incorrect text.

**KV Cache** — Storing computed Key and Value tensors from previous tokens to avoid recomputation during generation. Critical for inference speed.

**LoRA (Low-Rank Adaptation)** — Fine-tuning by adding small trainable matrices alongside frozen model weights. Much cheaper than full fine-tuning.

**MoE (Mixture of Experts)** — Architecture where only a subset of "expert" FFN layers activate per token. Mixtral, DeepSeek V3 use this. More total params but fewer active params = cheaper inference.

**MQA (Multi-Query Attention)** — All Q heads share a single K,V head. More aggressive memory savings than GQA.

**Pre-norm** — Applying LayerNorm *before* attention/FFN (not after). Used by GPT-2 onward. More stable training.

**Quantization** — Reducing weight precision (e.g., float16 → int4) to shrink model size and speed up inference. Some quality loss.

**RAG (Retrieval-Augmented Generation)** — Fetching relevant documents and including them in the prompt. Reduces hallucination.

**Residual Connection** — Adding the input of a layer back to its output (`x + layer(x)`). Lets gradients flow through deep networks.

**RoPE (Rotary Position Embedding)** — Encoding position by rotating the Q,K vectors. Relative (not absolute), and can extrapolate beyond training length.

**RMSNorm** — A simpler, faster version of LayerNorm that skips the mean computation. Used by Llama, Mistral.

**Scaling Laws** — Empirical relationships between model size, data, compute, and performance. "Chinchilla scaling" showed most models were undertrained.

**SwiGLU** — An activation function (Swish × Gate Linear Unit) used in modern LLMs. Replaces GELU from GPT-2.

**Temperature** — A generation parameter that controls randomness. Low (0.1) = deterministic, High (1.5) = creative/chaotic.

**Top-k / Top-p** — Sampling strategies that limit which tokens can be selected. Top-k: only top K tokens. Top-p (nucleus): tokens whose cumulative probability exceeds P.

**Transformer** — The architecture (Vaswani et al., 2017) that all modern LLMs are based on. Core idea: self-attention + feed-forward networks, stacked in layers.

**Weight Tying** — Using the same matrix for token embeddings and the output prediction head. Fewer parameters, better representations.
