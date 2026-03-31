# Contributing

Thanks for wanting to contribute! Here's how.

## What's Welcome

- **Bug fixes** in code experiments
- **Additional benchmarks** for existing models (with documented methodology)
- **Translations** of guides
- **New experiments** that extend an existing model's analysis

## Structure Rules

Every model directory must be self-contained:

```
models/XX-model-name/
├── README.md           ← Required. Overview + how to run
├── DEEP-DIVE.md        ← Required. Full technical analysis
├── code/
│   ├── requirements.txt
│   └── main.py         ← Must be runnable with: python main.py
├── notebooks/          ← Optional
└── diagrams/
    └── *.mmd           ← Mermaid source files (not just PNGs)
```

## Code Standards

- Every `code/` directory must have a `requirements.txt`
- `python main.py` must work with no arguments (sensible defaults)
- Include a `--help` flag
- No API keys committed — use environment variables
- Test on Python 3.10+

## Benchmarks

If you add benchmarks:
- Document hardware (CPU/GPU, memory)
- Document methodology (prompt, sample size, temperature)
- Include raw results, not just summaries
- Compare against at least one other model from this repo

## Pull Requests

- One model per PR
- Run the code before submitting
- Keep the README learner-friendly — explain *why*, not just *what*
