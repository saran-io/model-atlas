# AI Models from the Inside Out

### 24 weeks. 24 models. Every one built from scratch or benchmarked on real tasks.

I'm **Saran**, co-founder of [Tekvo](https://tekvo.ai), where we build agentic AI systems. This repo is my open notebook — every week I pick one AI model, tear it apart, build something real with it, and document everything so you can learn from it.

**This is not a paper summary repo.** Every model has:
- Runnable code you can execute in minutes
- Architecture diagrams you can actually understand
- Benchmarks I ran myself (not copy-pasted from leaderboards)
- A "When to use this" verdict from someone who ships AI products

---

## Quick Start

Pick a model that interests you and dive in. Each has its own README with setup instructions.

```bash
# Example: Run GPT-2 from scratch
cd models/01-gpt2-from-scratch/code
pip install -r requirements.txt
python gpt2.py --prompt "The future of AI is"
```

---

## The Models

### Phase 1 — Foundations & Frontier (Weeks 1-6)

| # | Model | What You'll Learn | Difficulty |
|---|-------|-------------------|------------|
| [01](models/01-gpt2-from-scratch) | **GPT-2 from scratch** | Transformers, self-attention, tokenization — by building it | ⭐⭐ |
| [02](models/02-gpt4o-vs-gpt41) | **GPT-4o vs GPT-4.1** | How frontier models differ on real agentic tasks | ⭐ |
| [03](models/03-claude-opus-sonnet) | **Claude Opus & Sonnet** | Tool use, constitutional AI, long context | ⭐ |
| [04](models/04-llama4) | **Llama 4 (Meta)** | Open weights, self-hosting, cost analysis | ⭐⭐ |
| [05](models/05-deepseek-r1) | **DeepSeek R1** | MoE architecture, reasoning at low cost | ⭐⭐ |
| [06](models/06-qwen) | **Qwen 2.5 / QwQ** | Multilingual, open-weight MoE | ⭐⭐ |

### Phase 2 — The Wider Landscape (Weeks 7-12)

| # | Model | What You'll Learn | Difficulty |
|---|-------|-------------------|------------|
| [07](models/07-gemini) | **Gemini 2.x** | Native multimodal, 1M+ context | ⭐ |
| [08](models/08-o3-reasoning) | **o3 / o4-mini** | Reasoning models, chain-of-thought scaling | ⭐⭐ |
| [09](models/09-mistral-mixtral) | **Mistral / Mixtral** | MoE routing, sparse vs dense trade-offs | ⭐⭐⭐ |
| [10](models/10-phi-small-models) | **Phi-4 / Small Models** | On-device deployment, distillation | ⭐⭐ |
| [11](models/11-cohere-command-r) | **Cohere Command R+** | RAG-optimized models, citation grounding | ⭐ |
| [12](models/12-tokenization-deep-dive) | **Tokenization Across Models** | BPE, SentencePiece, tiktoken — compared | ⭐⭐ |

### Phase 3 — Specialized Models (Weeks 13-18)

| # | Model | What You'll Learn | Difficulty |
|---|-------|-------------------|------------|
| [13](models/13-code-models) | **Code Models** | Codestral, StarCoder, DeepSeek-Coder compared | ⭐⭐ |
| [14](models/14-embedding-models) | **Embedding Models** | Voyage, BGE, E5 — when dimensions matter | ⭐⭐ |
| [15](models/15-vision-language) | **Vision-Language Models** | LLaVA, Qwen-VL, GPT-4V architectures | ⭐⭐⭐ |
| [16](models/16-audio-speech) | **Audio & Speech** | Whisper, Moshi — voice-to-agent pipelines | ⭐⭐ |
| [17](models/17-quantization) | **Quantization** | GGUF, GPTQ, AWQ — quality vs speed vs cost | ⭐⭐⭐ |
| [18](models/18-fine-tuning) | **Fine-tuning** | LoRA, QLoRA, RLHF in practice | ⭐⭐⭐ |

### Phase 4 — Production & Architecture (Weeks 19-24)

| # | Model | What You'll Learn | Difficulty |
|---|-------|-------------------|------------|
| [19](models/19-model-routing) | **Model Routing** | Pick the right model per task, automatically | ⭐⭐⭐ |
| [20](models/20-serving-infra) | **Serving Infrastructure** | vLLM, TGI, Triton — benchmarked | ⭐⭐⭐ |
| [21](models/21-multi-agent) | **Multi-Agent Systems** | Different models for different agent roles | ⭐⭐⭐ |
| [22](models/22-model-selection-guide) | **Model Selection Guide** | Decision framework from 22 weeks of building | ⭐⭐ |
| [23](models/23-cost-optimization) | **Cost Optimization** | Real pricing analysis across all models | ⭐⭐ |
| [24](models/24-the-complete-picture) | **The Complete Architecture** | How everything fits together | ⭐⭐⭐ |

---

## How Each Model is Structured

Every model directory follows the same pattern:

```
models/01-gpt2-from-scratch/
├── README.md              ← Start here. Overview, key insights, how to run
├── DEEP-DIVE.md           ← Full technical write-up with architecture analysis
├── code/
│   ├── requirements.txt   ← Dependencies
│   ├── main.py            ← The primary experiment (always runnable)
│   └── ...                ← Supporting code
├── notebooks/
│   └── explore.ipynb      ← Interactive exploration (optional)
└── diagrams/
    ├── architecture.mmd   ← Mermaid source (version controlled)
    └── architecture.png   ← Rendered diagram
```

---

## How to Use This Repo

### 🔰 If you're learning AI/ML
Go in order, starting from [01-gpt2-from-scratch](models/01-gpt2-from-scratch). Each model builds on concepts from the previous ones. Run the code — reading is not enough.

### 🛠 If you're building AI products
Jump to the model you're evaluating. Check the "When to use / When to skip" section in each README. The benchmarks use real agentic tasks, not toy examples.

### 🏗 If you're designing AI systems
Start with [19-model-routing](models/19-model-routing) and [21-multi-agent](models/21-multi-agent). These cover how to combine multiple models in production architectures.

---

## Guides

Standalone reference guides that span multiple models:

| Guide | Description |
|-------|-------------|
| [Model Comparison Matrix](guides/model-comparison-matrix.md) | Side-by-side specs, pricing, and benchmarks (updated weekly) |
| [Which Model Should I Use?](guides/which-model.md) | Decision flowchart for common use cases |
| [Glossary](guides/glossary.md) | Transformer jargon explained in plain english |

---

## Run Locally

```bash
git clone https://github.com/saran-io/model-atlas.git
cd model-atlas

# Each model has its own requirements.txt
cd models/01-gpt2-from-scratch/code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Most models run on a MacBook. GPU-heavy experiments note this in their README.

---

## Follow the Journey

New model every Friday.

- **Blog**: [saran-io.github.io/blog](https://saran-io.github.io/blog)
- **LinkedIn**: [linkedin.com/in/saran-io](https://www.linkedin.com/in/saran-io/)
- **X**: [@saran_io](https://x.com/saran_io)

---

## Contributing

Found a bug? Have a better benchmark? PRs welcome.
- Keep code self-contained per model directory
- Include a README explaining what your code does
- Benchmarks need methodology documented

## License

Code: MIT | Written content: CC BY 4.0
