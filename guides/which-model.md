# Which AI Model Should I Use?

> A decision framework built from 24 weeks of building with every major model.
> Updated as new models are tested.

## Quick Decision Table

| Your Task | Best Choice | Runner-up | Why |
|-----------|------------|-----------|-----|
| **General chat/assistant** | GPT-4o | Claude Sonnet | Best all-around quality/speed |
| **Complex reasoning** | o3 / DeepSeek R1 | Claude Opus | Chain-of-thought, multi-step logic |
| **Long document analysis** | Gemini 2.x | Claude (200K) | Gemini handles 1M+ tokens natively |
| **Code generation** | Claude Sonnet | GPT-4.1 | Best at understanding intent + correctness |
| **Agentic tool use** | Claude Sonnet/Opus | GPT-4.1 | Structured output, reliable function calling |
| **RAG pipeline** | Cohere Command R+ | GPT-4o | Built-in citation grounding |
| **Multilingual** | Qwen 2.5 | GPT-4o | Best non-English performance, especially CJK |
| **Self-hosted (cost)** | Llama 4 / Qwen 2.5 | Mistral | Open weights, strong community |
| **Self-hosted (quality)** | DeepSeek R1 | Llama 4 70B | Closest to frontier at open-weight |
| **On-device / mobile** | Phi-4 / Gemma | Qwen 2.5 3B | Small models, optimized for edge |
| **Embeddings** | Voyage 3 | Cohere Embed v3 | Best retrieval quality per dimension |
| **Vision + text** | GPT-4o | Gemini 2.x | Native multimodal understanding |
| **Batch processing (cheap)** | DeepSeek V3 | GPT-4o-mini | Lowest cost per million tokens |

## Decision Flowchart

```
START: What are you building?
│
├─ An agent that uses tools?
│  ├─ Needs complex reasoning? → Claude Opus / o3
│  ├─ Needs speed + cost efficiency? → Claude Sonnet / GPT-4o-mini
│  └─ Self-hosted requirement? → Llama 4 + function calling
│
├─ A RAG system?
│  ├─ Need citations? → Cohere Command R+
│  ├─ Large doc corpus? → Gemini (long context) + Voyage (embeddings)
│  └─ Budget constrained? → Qwen 2.5 + BGE embeddings
│
├─ Content generation?
│  ├─ Marketing/creative → GPT-4o (best fluency)
│  ├─ Technical writing → Claude (best at following complex instructions)
│  └─ Code documentation → Claude Sonnet
│
├─ Code assistance?
│  ├─ IDE integration → Claude Sonnet / GPT-4.1
│  ├─ Code review → Claude Opus (best at catching subtle bugs)
│  └─ Self-hosted → DeepSeek-Coder / StarCoder
│
└─ Data processing / analysis?
   ├─ Structured extraction → GPT-4o (best JSON mode)
   ├─ Classification at scale → GPT-4o-mini / Haiku (cheapest)
   └─ Multilingual data → Qwen 2.5
```

## Cost vs Quality Trade-offs

*Rough positioning — check each model's README for exact benchmarks*

```
                        High Quality
                            │
               Claude Opus ●│● o3
                            │
            GPT-4o ●  ● Claude Sonnet  ● DeepSeek R1
                            │
     Gemini Flash ●   ● GPT-4o-mini    ● Llama 4 70B
                            │
         Haiku ●      ● Qwen 2.5 72B   ● Mixtral 8x22B
                            │
  ──────────────────────────┼──────────────────────────
  Low Cost                  │                High Cost
                            │
                       Low Quality
```

---

*This guide evolves weekly as I test each model. See individual model directories for detailed benchmarks.*
