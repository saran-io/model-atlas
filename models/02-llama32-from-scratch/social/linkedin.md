I just shipped Week 2 of Model Atlas: Llama 3.2 From Scratch.

Week 1 was GPT-2: the cleanest way to understand decoder-only transformers.

Week 2 asks the natural next question:

What did modern open LLMs actually change?

The answer is not "a totally new architecture." Llama is still:

tokens -> embeddings -> decoder blocks -> logits

But the block got upgraded:

- LayerNorm -> RMSNorm
- learned position embeddings -> RoPE
- multi-head attention -> grouped-query attention
- GELU MLP -> SwiGLU

These sound like implementation details until you care about real systems.

RoPE helps with long-context behavior.
GQA reduces KV-cache memory during inference.
RMSNorm is simpler and cheaper.
SwiGLU gives the feed-forward layer a stronger gated path.

That is the theme of the series: strip down one model each week, build the core idea, then explain where it matters in production.

Code and diagrams are in the repo:
https://github.com/saran-io/model-atlas
