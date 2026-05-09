# Model Atlas

**One AI model each week, built from the inside out.**

I'm **Saran**, co-founder of [Tekvo](https://tekvo.ai), where we build agentic AI systems. This repo is my open notebook for learning modern AI models by rebuilding the important pieces, drawing the architecture, and explaining where each model actually fits in production.

This is not a paper-summary repo. Each week tries to answer five practical questions:

1. What problem did this model solve?
2. What is the key architecture idea?
3. Can we build the minimal version in code?
4. How is it different from the previous model?
5. Where would I use or avoid it in a real product?

## What We Do

Every model deep dive includes:

- **Build:** a small, runnable implementation or experiment.
- **Visualize:** clean architecture diagrams in version-controlled source files.
- **Compare:** what changed from earlier models and why it matters.
- **Apply:** product and system-design notes from an AI builder's point of view.
- **Run:** commands so readers can reproduce the core experiment locally.
- **Remember:** the few ideas that should stick after reading.

The goal is to make model architecture feel physical: tensors, blocks, caches, tradeoffs, and use cases, not just benchmark tables.

## Start Here

If you are new to the repo, go in order:

| Week | Model | Status | What it teaches |
|---|---|---|---|
| [01](models/01-gpt2-from-scratch) | GPT-2 from scratch | Published | Decoder-only transformers, causal attention, tokenization, weight loading |
| [02](models/02-llama32-from-scratch) | Llama 3.2-style decoder | Drafted | RMSNorm, RoPE, grouped-query attention, SwiGLU, modern LLM blocks |

Quick run:

```bash
git clone https://github.com/saran-io/model-atlas.git
cd model-atlas

# Week 1: GPT-2
cd models/01-gpt2-from-scratch/code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python gpt2.py --prompt "The future of AI is" --max_tokens 80
```

Each model has its own `requirements.txt`, so install dependencies inside that model's `code/` folder.

## Current Learning Path

The first arc is about architecture evolution: start with the clean GPT-2 transformer, then move toward the design choices used in modern open models.

| Week | Model / Topic | Build project | Key idea |
|---|---|---|---|
| 1 | GPT-2 | Implement GPT-2 in PyTorch and load real weights | The baseline decoder-only transformer |
| 2 | Llama 3.2 | Implement a Llama-style decoder block | How modern LLMs improved the GPT-2 block |
| 3 | Mistral / Mixtral | Visualize dense vs sparse MoE inference | Why models can grow without activating every parameter |
| 4 | DeepSeek V3 | Study large-scale MoE and efficiency choices | Training and serving tradeoffs at frontier scale |
| 5 | DeepSeek R1 | Compare reasoning behavior and post-training | Why reasoning models feel different |
| 6 | Qwen | Build multilingual and coding evaluations | Strong open models beyond the Western default set |
| 7 | Phi / Gemma | Run small-model local experiments | When small models are the right product choice |
| 8 | BERT | Build encoder-only text tasks | Search, classification, and embeddings before chatbots |

The full working roadmap lives in [ROADMAP.md](ROADMAP.md).

## Repository Structure

```text
model-atlas/
├── blog/                         # Publish-ready blog drafts
├── guides/                       # Cross-model references and decision guides
├── models/
│   ├── 01-gpt2-from-scratch/
│   │   ├── README.md             # Week overview and quick start
│   │   ├── DEEP-DIVE.md          # Longer technical notes when available
│   │   ├── code/                 # Runnable implementation or benchmark
│   │   ├── diagrams/             # Mermaid / Excalidraw diagram sources
│   │   └── social/               # LinkedIn and X drafts
│   └── 02-llama32-from-scratch/
├── ROADMAP.md                    # 24-week content plan
└── README.md
```

## How To Use This Repo

If you are learning AI/ML, start at Week 1 and run the code. The posts are written so each model adds one or two new ideas on top of the previous week.

If you are building AI products, jump to the model or system pattern you are evaluating and look for the "where this applies" sections. The practical question is always: what does this architecture make easier, cheaper, or more reliable?

If you are designing AI systems, follow the comparison diagrams. They are meant to show why choices like RoPE, GQA, MoE, quantization, and routing matter once models become part of a real product.

## Guides

Standalone references that grow as the series grows:

| Guide | Description |
|---|---|
| [Model Comparison Matrix](guides/model-comparison-matrix.md) | Side-by-side specs, pricing, and benchmark notes |
| [Which Model Should I Use?](guides/which-model.md) | Decision flowchart for common use cases |
| [Glossary](guides/glossary.md) | Transformer and AI-system terms explained plainly |

## Follow Along

New model deep dives are planned weekly.

- **Blog:** [saran-io.github.io/blog](https://saran-io.github.io/blog)
- **LinkedIn:** [linkedin.com/in/saran-io](https://www.linkedin.com/in/saran-io/)
- **X:** [@saran_io](https://x.com/saran_io)

## Contributing

Corrections, clearer diagrams, benchmark improvements, and reproducibility fixes are welcome.

- Keep each model self-contained under `models/XX-model-name/`.
- Include a README for any new experiment.
- Document benchmark methodology before adding results.
- Prefer small runnable examples over large hidden setup.

## License

Code: MIT. Written content: CC BY 4.0.
