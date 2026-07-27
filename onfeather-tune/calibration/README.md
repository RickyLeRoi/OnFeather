# Calibration data

Measurements that turn the cost model from a guess into a prediction.

The model `of tune` plans against is:

```
decode tok/s  =  effective read bandwidth  /  active bytes per token
```

Both terms are computable — the first from `of probe`, the second from the GGUF
header — but converting a benchmark number into *effective* bandwidth needs a
constant, and a constant needs measurements. That is what lives here.

## How to contribute a run

The most useful contribution to this repository. You need a machine and a model
you already have.

```bash
of probe --json -o probe.json            # your hardware
of inspect model.gguf --json             # active bytes per token
```

Then measure real decode speed on the same model. Via Ollama:

```bash
curl -s http://localhost:11434/api/generate -d '{
  "model": "your-model", "prompt": "Count from one to fifty.",
  "stream": false, "options": {"num_predict": 128, "temperature": 0}
}' | jq '.eval_count / (.eval_duration / 1e9)'
```

Or via llama.cpp directly:

```bash
llama-bench -m model.gguf -p 0 -n 128 -o json
```

Open a PR adding a JSON file named `<date>-<cpu-slug>.json`, following the shape
of the existing entries. Please include:

- **At least two models of clearly different sizes.** One run cannot separate a
  correct bandwidth model from a coincidence; two that agree can.
- **Generated tokens, not just tok/s.** Short samples are noisy — 100+ tokens.
- **Which backend actually ran.** `ollama ps` reports the CPU/GPU split, and a
  partially offloaded run measures something else entirely.
- **A cold-start caveat if relevant.** Model load time must be excluded.

## What the current data says

| Date | CPU | Memory | STREAM add | Implied read | Factor |
|---|---|---|---|---|---|
| 2026-07-25 | i9-9880H | DDR4-2666 dual | 20.0 GB/s | 24.6–25.1 GB/s | **1.25** |
| 2026-07-26 | EPYC 7B12 (18 of 64 cores) | server, shared host | 74.5 GB/s | 56.7 GB/s | **0.76** |

**The factor is not universal.** That is the single most important line in this
directory, and it took a second machine to learn it. A laptop reaches 1.25×
STREAM; a rented server reached 0.76×. The server was shared and its bandwidth
readings were unstable — 43.8 and 25.2 GB/s for the same nominal measurement —
so one run does not explain *why*. It is enough to establish *that*, and to make
the registry the point rather than an accessory.

The prediction itself still held on that machine: `Qwen3-30B-A3B-Q4_K_M` was
predicted at 26.7 tok/s (range 22.7–29.4) and measured **23.64**, inside the
range, on hardware sharing nothing with the calibration host.

The bandwidth-bound premise held too. Decode gained 6% between 8 and 18 threads
— waiting on memory, not arithmetic — then collapsed past the container's
entitlement: 26.4 tok/s at 18 threads, 12.8 at 36, 6.7 at 64.

Machine one, three models:

| Model | Architecture | Active/token | Predicted | Measured | Error |
|---|---|---|---|---|---|
| qwen2.5-coder 7B | qwen2, dense | 4.68 GB | 5.35 t/s | 5.35 t/s | −0.1% |
| qwen2.5-coder 14B | qwen2, dense | 8.98 GB | 2.78 t/s | 2.79 t/s | −0.2% |
| qwen3.6 27B | qwen35, hybrid SSM + vision | 16.22 GB | 1.54 t/s | 1.51 t/s | +1.8% |

A 3.5× span of active weight and two unrelated architectures land on the same
implied bandwidth within 2%. The 27B matters most: it is a hybrid state-space
model with a vision encoder and a speculative-decoding head, none of which
resemble the two dense transformers — and the prediction still holds, because
the model is about bandwidth rather than about transformers.

The factor exceeds 1.0 because streaming weights is pure sequential read, while
the STREAM `add` kernel writes one array for every two it reads and pays
read-for-ownership on the writes.

**What is not yet known:** what the factor is on DDR5, on Apple Silicon's unified
memory, on quad-channel workstations, or on AVX-512 and AMX machines whose
dequantisation cost differs — and, now that two machines disagree by 40%, what
actually drives the difference. Contention, core count, NUMA topology and memory
channel count are all candidates and none is ruled out. Every one of those is a
PR someone could send.

**If you run this in a container** — which includes every rented GPU — check
that `of probe` reports a *Usable* core count and a cgroup *Limit* where they
apply. Benchmarking with more threads than the container is entitled to
measured a third of the real bandwidth, which would poison any run contributed
from such a machine.
