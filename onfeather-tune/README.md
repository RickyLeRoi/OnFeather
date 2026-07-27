# OnFeather — `of tune`

**Run heavy models on light hardware.**

Your GPU has 4 GB of VRAM. The model you want is 30 B parameters. Conventional
wisdom says no — but if the model is a Mixture-of-Experts, only a fraction of its
weights are active per token, and the right split puts attention and the KV cache
on the GPU while the expert FFNs stream from system RAM.

llama.cpp can already do this:

```bash
llama-server -m qwen3-30b-a3b-q4_k_m.gguf -ngl 99 \
  -ot "blk\.\d+\.ffn_(gate|up|down)_exps\.weight=CPU"
```

The problem is that nobody knows how to write that regex, and the *optimal* split
changes with the model, the quantisation, the context length and the machine.
Get it right and the model is usable. Get it wrong and you get 2 tok/s, or an
out-of-memory crash.

`of tune` works out the right split for you.

## Status

Pre-alpha, but the planning path works end to end.

| Command | Status | What it does |
|---|---|---|
| `of probe` | ✅ working | Profiles the machine: cores, ISA, **measured** memory bandwidth, GPUs |
| `of inspect` | ✅ working | Breaks a GGUF down by tensor role, estimates speed on this machine |
| `of plan` | ✅ working | Computes the offload split and emits the `llama-server` command |
| `of tune` | 🚧 next | Benchmarks the top candidate plans, elects a winner |
| `of serve` | 📋 planned | Launches `llama-server` with the winning plan |

The planning path is now measured end to end on CUDA — see
[does it actually work?](#does-it-actually-work) below.

The cost model predicts measured decode speed to within 1.8% **while memory
bandwidth is the bottleneck** — CPU-only, or with most of the model streaming
from RAM. Push enough onto the GPU and it runs away: with a tenth of the active
weights crossing the bus it predicted 71 tok/s against a measured 33. `of plan`
now says so rather than printing the number bare. The placement it picks is
still the right one; the speed it promises for it is not.

A second machine, a rented AMD EPYC server, landed inside the predicted range on
the first try — and showed the model's central constant is **not universal**,
which is the whole argument for the registry below.

Three machines are recorded in [calibration/](calibration/); please send a run
from yours. [docs/cuda-validation.md](docs/cuda-validation.md) is the recipe, and it
starts on a free notebook — which is where every result on this page came from,
after two rented pods failed to initialise CUDA at all.

## Does it actually work?

Measured on a Tesla P100 (16 GiB) running `Qwen3-30B-A3B-Q4_K_M`, a 17.3 GiB MoE
that does not fit — `-ngl 99` alone refuses to load it. Context 8192.

| Configuration | Decode | |
|---|---|---|
| CPU only | 6.79 tok/s | the starting point |
| 4 expert layers on CPU | — | **will not load** |
| 6 expert layers on CPU | — | **out of memory** |
| **9 — what `of plan` computed** | **32.87 tok/s** | **4.8× the CPU baseline** |
| 12 (a cautious guess) | 28.63 tok/s | −13%, for ever |
| 24 (a very cautious guess) | 18.34 tok/s | −44% |

The plan is the fastest configuration that runs at all. **One step more
aggressive crashes; three steps more cautious costs 13%.** That narrow band is
the argument for computing the number instead of guessing it, and the numbers
above are what it costs to be wrong in either direction.

Two caveats worth stating. The plan leaves 0.04 GiB of headroom, so it is
optimal and *fragile* — a different build or allocator could tip it into an OOM,
and `--reserve` is the dial for trading speed against that risk. And the same
comparison at a ~640-token context put a 6-layer guess 18% ahead, because a plan
computed for 8192 reserves KV cache a short run never uses: **the plan is only
right for the context you tell it about**.

## Isn't this what `--n-cpu-moe` already does?

For this model, the generated command *is* `--n-cpu-moe 9`: both were measured
at 32.87 and 32.91 tok/s, agreeing to 0.2%. The placement `of plan` produces is
a contiguous suffix of layers, which is exactly what that flag expresses.

So the honest claim is not that `of tune` reaches placements the flag cannot
describe — today it does not. It is that **`--n-cpu-moe` makes you supply the
number**, and the table above shows what supplying it badly costs. `of tune`
derives it from your measured VRAM, the KV cache your context actually needs,
and the size of the tensors that must stay resident.

`--fit` auto-shrinks until a configuration stops crashing, which lands you
somewhere safe rather than somewhere fast — the 12 and 24 rows are what "safe"
looks like.

Finer-grained placement — per tensor type, or non-contiguous layers chosen by
expert traffic — is a plausible next step and an unproven one. It is not what
this tool does yet.

## Install

```bash
git clone https://github.com/RickyLeRoi/onfeather-tune
cd onfeather-tune
pip install -e .
```

## Usage

```bash
of probe              # human-readable summary
of probe --json       # machine-readable profile
of probe -o hw.json   # save it
```

```
CPU
  Model                Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz
  Cores                8 physical / 16 logical
  ISA                  avx2, f16c, fma

MEMORY
  Total                32.0 GiB
  Bandwidth (1 thread) 12.3 GB/s
  Bandwidth (all)      20.0 GB/s

GPU
  Name                 AMD Radeon Pro 5500M (amd)
  Memory               4.0 GiB
```

```bash
of inspect model.gguf     # what is in it, and how fast it can go here
of plan model.gguf        # where every tensor should live
of plan model.gguf --context 8192 --reserve 768
```

A 30 B MoE planned for a 4 GB laptop GPU:

```
PLAN
  Hot tensors            GPU
  Expert layers on GPU   3 of 48
  Expert layers on CPU   45 of 48

VRAM BUDGET
  Available              3.50 GiB
  Hot tensors            0.69 GiB
  KV cache               1.50 GiB @ 32768
  Experts kept           1.09 GiB
  Headroom               0.21 GiB

PROJECTED
  Read from RAM/token    1.03 GiB
  Decode speed           22.3 tok/s (18.9-24.5)

COMMAND
  llama-server -m model.gguf -ngl 99 -ot "blk\.(3|4|...|47)\.ffn_(gate|up|down)_exps\.weight=CPU" -c 32768
```

The model is 18 GiB and the GPU holds 4. Naively pushing every expert to system
RAM reads 1.8 GiB per token; paying for the hot tensors first and spending what
is left on experts reads 1.03 GiB — roughly twice the speed, from placement alone.

### Does the model actually predict anything?

On the calibration host, yes, to within 1.8%:

| Model | Architecture | Active per token | Predicted | Measured |
|---|---|---|---|---|
| qwen2.5-coder 7B | qwen2, dense | 4.68 GB | 5.35 t/s | **5.35 t/s** |
| qwen2.5-coder 14B | qwen2, dense | 8.98 GB | 2.78 t/s | **2.79 t/s** |
| qwen3.6 27B | qwen35, hybrid SSM + vision | 16.22 GB | 1.54 t/s | **1.51 t/s** |

The three runs independently imply 24.6–25.1 GB/s of read bandwidth — agreement
within 2% across a 3.5x range of size and two unrelated architectures. The 27B
matters most: it is a hybrid state-space model with a vision encoder and a
speculative-decoding head, nothing like the two dense transformers, and the
prediction still holds — because the model is about bandwidth rather than about
transformers.

It also corrected a real error. The first version assumed inference would reach
some fraction *below* STREAM bandwidth; it actually reaches **1.25×** it, because
streaming weights is pure sequential read while the STREAM `add` kernel writes
one array for every two it reads and pays read-for-ownership on the writes. The
original guess was pessimistic by nearly 2×.

### A second machine, and the limit of a single constant

A rented AMD EPYC 7B12 with an RTX 3070 ran `Qwen3-30B-A3B-Q4_K_M`, a 30 B MoE
nothing like the calibration models:

| | |
|---|---|
| Predicted | **26.7 tok/s** (22.7–29.4) |
| Measured | **23.64 tok/s** |

Inside the range, first attempt, on hardware sharing nothing with the machine
the model was built on. Two things came out of that run:

**The bandwidth-bound premise holds.** Decode gains 6% between 8 and 18 threads
— it is waiting on memory, not on arithmetic, which is the assumption every
placement decision rests on. Push past the machine's entitlement and it
collapses:

| Threads | Decode |
|---|---|
| 8 | 24.8 tok/s |
| 18 | **26.4 tok/s** |
| 36 | 12.8 tok/s |
| 64 | 6.7 tok/s |

**`WEIGHT_READ_FACTOR = 1.25` does not transfer.** On that server the same
calculation gives **0.76**. The host was shared and its bandwidth readings were
unstable, so one measurement does not settle the cause — but it is enough to
retire the idea that one constant describes every machine. A registry of real
measurements is the answer; a better constant is not.

### Containers, where rented GPUs live

Both bugs found by running `of probe` on a rented pod were the same bug in two
places: it described the host instead of the container.

- It reported **251.5 GiB** of RAM against a real cgroup limit of **32.6 GiB**.
- It reported **128 cores** against an entitlement of **18** — and llama.cpp's
  default thread count would have followed the larger number, at a **4× decode
  penalty** nothing would have reported.

`of probe` now reads cgroup limits and the affinity mask, benchmarks with the
cores it is actually allowed, and `of plan` emits an explicit `-t` when the two
disagree. On ordinary hardware nothing changes and no `-t` appears.

### Hot and cold

Everything rests on one asymmetry. In a Mixture-of-Experts, most of the file is
routed experts, of which only a handful fire per token:

| Tensor | Runs on | Verdict |
|---|---|---|
| `attn_*`, `*_norm`, `token_embd`, `output` | every token | **hot** — keep on GPU |
| `ffn_gate_inp` (the router) | every token | **hot** — it *picks* the experts |
| `ffn_*_shexp` (shared experts) | every token | **hot** |
| `ffn_*_exps` (routed experts) | ~6 % of tokens each | **cold** — offload these |

Confusing a shared expert (`_shexp`) with a routed one (`_exps`) is the easiest
way to destroy performance, and nothing warns you: the run just gets slow. The
generated regex targets `_exps` and nothing else.

### Why bandwidth is measured, not looked up

CPU-side inference of offloaded experts is **bandwidth-bound**, not FLOP-bound:
every token drags the active experts across the memory bus. Decode speed is
roughly

```
tok/s  ≈  memory bandwidth  /  active bytes per token
```

which makes measured bandwidth the single best predictor of how a plan will
perform — and the reason `of probe` benchmarks it instead of reading the spec
sheet. The same DDR5-5600 kit delivers half its rated bandwidth in
single-channel, and no amount of `dmidecode` will tell you that.

## The plan

The part that makes this worth building is the **community registry**: a
`hardware fingerprint → best known plan` mapping, published as plain JSON in this
repo. Run `of probe`, and if someone with your hardware has already tuned your
model, you download their answer instead of spending forty minutes benchmarking.
The registry gets better as more people use it.

This started as a convenience and the second calibration turned it into the
point. If the read factor were universal, a formula would be enough and the
registry would be a cache. It is not universal — 1.25 on a laptop, 0.76 on a
server — so measurements from real machines are the only thing that can be
trusted, and collecting them is the product.

`of probe` already emits the fingerprint — a stable, bucketed hash of the
attributes that actually change the optimal plan.

## Related work

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — the engine. `of tune`
  drives it as a subprocess rather than binding to it, so upstream changes to
  kernels and quant formats come for free.
- [ktransformers](https://github.com/kvcache-ai/ktransformers) — heterogeneous
  CPU/GPU MoE inference with hand-tuned kernels. Faster where it applies;
  harder to install and narrower in model support.

`of tune` is not another inference engine. **The product is the decision**: given
*this* machine and *this* GGUF, what is the optimal configuration?

## Licence

Apache-2.0
