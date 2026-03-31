# Model Comparison Matrix

> Updated weekly as each model is benchmarked. Blanks = not yet tested.

## Specs

| Model | Params | Architecture | Context | Open Weights | License |
|-------|--------|-------------|---------|:------------:|---------|
| GPT-2 | 124M | Dense Transformer | 1K | Yes | MIT |
| GPT-4o | ~1.8T (est.) | Dense (est.) | 128K | No | Proprietary |
| GPT-4.1 | ~1.8T (est.) | Dense (est.) | 1M | No | Proprietary |
| Claude Opus | Undisclosed | Undisclosed | 200K | No | Proprietary |
| Claude Sonnet | Undisclosed | Undisclosed | 200K | No | Proprietary |
| Gemini 2.x | Undisclosed | MoE (est.) | 1M+ | No | Proprietary |
| o3 | Undisclosed | Reasoning + Transformer | 200K | No | Proprietary |
| Llama 4 | 8B–405B | Dense / MoE | 128K+ | Yes | Llama License |
| Mistral/Mixtral | 7B–8x22B | Dense / MoE | 32K | Yes | Apache 2.0 |
| Qwen 2.5 | 0.5B–72B | Dense / MoE | 128K | Yes | Apache 2.0 |
| DeepSeek V3 | 671B (37B active) | MoE | 128K | Yes | DeepSeek License |
| DeepSeek R1 | 671B (37B active) | MoE + Reasoning | 128K | Yes | MIT |
| Phi-4 | 14B | Dense | 16K | Yes | MIT |
| Cohere Command R+ | Undisclosed | Dense | 128K | Partial | Proprietary |

## Pricing (per 1M tokens, as of March 2026)

| Model | Input | Output | Notes |
|-------|-------|--------|-------|
| GPT-4o | $2.50 | $10.00 | |
| GPT-4.1 | $2.00 | $8.00 | |
| Claude Opus | $15.00 | $75.00 | |
| Claude Sonnet | $3.00 | $15.00 | |
| Gemini 2.0 Flash | $0.10 | $0.40 | |
| o3 | $10.00 | $40.00 | Reasoning tokens add cost |
| DeepSeek V3 | $0.27 | $1.10 | API pricing |
| DeepSeek R1 | $0.55 | $2.19 | API pricing |
| Llama 4 | Free | Free | Self-hosted, pay for compute |
| Qwen 2.5 | Free | Free | Self-hosted, pay for compute |

*Prices change frequently. Check provider docs for current rates.*

## My Benchmarks (Filled In As Tested)

| Model | Agentic Tasks | Code Gen | Reasoning | Latency (TTFT) | Week |
|-------|:------------:|:--------:|:---------:|:--------------:|:----:|
| GPT-2 | N/A | N/A | N/A | <100ms | 01 |
| GPT-4o | — | — | — | — | 02 |
| Claude Sonnet | — | — | — | — | 03 |
| ... | | | | | |

*Detailed methodology in each model's directory.*
