# OnFeather

Three tools for running useful LLM work on hardware and budgets that shouldn't
allow it. They share nothing but a philosophy: local first, nothing hidden, no
service you have to pay to keep running.

Each project stands alone — install one without the others.

## [onfeather-tune](onfeather-tune/) — `of`

**Run heavy models on light hardware.**

Your GPU has 4 GB of VRAM and the model you want is 30 B parameters. If it's a
Mixture-of-Experts, the right split puts attention and the KV cache on the GPU
while the expert FFNs stream from RAM — and llama.cpp can already do that, if
you know the regex. `of` profiles your machine, reads the GGUF, and works out
the split for you.

→ [README](onfeather-tune/README.md)

## [onfeather-free](onfeather-free/) — `of-free`

**Aggregate free LLM tiers behind one quota-aware router.**

Groq, Google AI Studio, Cerebras, Mistral, GitHub Models and OpenRouter all run
real free tiers, each with its own limits and reset clocks. `of-free` keeps the
books and routes each request to whichever provider can still serve it, behind
an OpenAI-compatible endpoint any client can point at.

→ [README](onfeather-free/README.md)

## [onfeather-solo](onfeather-solo/) — `of-solo`

**A second brain whose memory you can read, correct and version.**

Most personal-memory tools store what they concluded about you as opaque
embeddings you can't inspect or argue with. `of-solo` stores memory as plain
Markdown with YAML frontmatter — one fact per file, in a directory you own, that
you can edit in any editor and put in git.

→ [README](onfeather-solo/README.md) · [SECURITY](onfeather-solo/SECURITY.md)

## Docker

All three ship in one image, published to GHCR:

```sh
cp .env.example .env      # API keys, all optional
docker compose up -d      # router on http://localhost:4141/v1
```

→ [DOCKER.md](DOCKER.md)

## Status

Pre-alpha, all three. The paths marked working in each README are tested; the
rest is honest about not existing yet.

Apache-2.0.
