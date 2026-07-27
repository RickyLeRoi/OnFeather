# CUDA validation runbook

**Goal:** prove the offload planner works on NVIDIA hardware, and produce a
second calibration point from a machine that is not the one it was built on.

**Budget:** about €1 of GPU time — but the minimum credit top-up, not the GPU,
is the real floor. See [what it actually costs](#what-it-actually-costs).

**Time:** about an hour, most of it spent compiling.

---

## Why this is needed

The cost model in `of tune` predicts decode speed to within 1.8% — on exactly
one machine: an Intel MacBook, CPU only, DDR4, no CUDA anywhere. Two things
remain unproven:

1. **The `-ot` regex path itself.** Every community datapoint for MoE expert
   offloading is CUDA. The generated command has never run against a real CUDA
   backend, so a wrong buffer-type name or a regex that llama.cpp's `std::regex`
   parses differently would be invisible so far.
2. **`WEIGHT_READ_FACTOR = 1.25`.** Calibrated against DDR4-2666 with AVX2. A
   machine with different memory, different kernels or a GPU in the loop has no
   reason to land on the same constant.

No private data is involved: a public GGUF and synthetic prompts. This is the
one task in the OnFeather toolkit that belongs on rented hardware.

## Step 0: do the valuable half for free

The single most valuable outcome of this exercise — *does the generated `-ot`
command even parse?* — needs no measurement fidelity whatsoever. It needs a CUDA
build of llama.cpp and one benchmark that either starts or errors out.

That runs on a free notebook:

| Service | GPU | System RAM | Session | Catch |
|---|---|---|---|---|
| **Kaggle Notebooks** | P100 16 GB or 2×T4 | ~29 GB | ~9 h, ~30 h/week | Phone verification to unlock GPU and internet |
| **Google Colab** (free) | T4 16 GB | ~13 GB | a few hours, throttled | Host RAM is smaller than the model |

Kaggle wins here for one reason that has nothing to do with the GPU: **system
RAM**. The offloaded experts have to live somewhere, and 13 GB of host RAM
cannot hold 18 GB of them — Colab free would fall back to mmap from disk and you
would be benchmarking the filesystem. Colab still works with a smaller MoE or a
Q3/IQ2 quant, but at that point you have changed the thing under test.

Both give you root inside the VM, so steps 2–8 below run unmodified in a
notebook cell prefixed with `!`. Enable the GPU accelerator, and on Kaggle turn
on *Internet* in the sidebar or every download fails silently late.

**If the `-ot` command errors here, stop.** You have the finding, it cost
nothing, and there is nothing worth calibrating until it is fixed.

Rent hardware for the other half: a decode-speed number attributable to known
hardware, on a card chosen to make the model not fit.

## Where to rent

| | RunPod | Vast.ai |
|---|---|---|
| Model | Datacenter (Secure Cloud) plus peer-hosted (Community Cloud) | Peer-hosted marketplace |
| Small-GPU price | roughly $0.15–0.25/h for a 3060/A4000 class card | often around half that |
| Minimum credit | ~$10 | ~$5 |
| Friction | Low — pick a template, click *Web Terminal* | Higher — pick an image, SSH key mandatory |
| Surprises | Few | Host-dependent disk and download speed, occasionally billed egress |

Prices move; treat these as an order of magnitude and read the live listing.
**RunPod is the recommendation for a one-off hour**: the browser terminal means
you never configure SSH, and its templates already carry the CUDA toolkit.
Vast.ai is cheaper and worth the setup if you expect to come back.

Both are prepaid, and neither has a free tier — which is what step 0 is for.

## What to rent

The GPU is the *least* important line on the listing. In order of what will
actually ruin the run:

| Spec | Requirement | Why |
|---|---|---|
| **System RAM** | **≥ 32 GB** | The offloaded experts live here. Below the model size llama.cpp mmaps from disk and you measure the SSD |
| **vCPU** | **≥ 8**, more is better | Expert FFNs are computed on the CPU. This is the half being calibrated |
| **Disk** | **≥ 60 GB** | 18 GB model, ~10 GB build tree, toolchain — against a 20 GB default |
| **VRAM** | **8–16 GB** — or 24 GB used as a [sweep rig](#a-big-card-is-a-sweep-rig-not-a-waste) | A 24 GB card fits the whole 30 B MoE, so its default plan offloads nothing |
| GPU model | RTX 3070 8 GB, RTX 3060 12 GB, A4000 16 GB, RTX 2000/4000 Ada | Any of them; cheapest wins |

The ordering is deliberate and it is not the one listing pages sort by. **VRAM
is the least important line**, because the premise of the test is a model that
does not fit: 18 GB fits in neither 8 GB nor 16 GB, so both cards exercise the
same path and the larger one buys nothing. Cores and system RAM decide the
quality of the calibration point; VRAM only decides how much of the plan lands
on the GPU.

**And if small cards are not available, take a large one.** `--reserve` withholds
VRAM from the budget, so any card can be made to produce a small card's plan —
see [below](#a-big-card-is-a-sweep-rig-not-a-waste). Availability should never
be what blocks this test.

There is also nothing to filter for called "CUDA": it comes from the host's
NVIDIA driver, so every NVIDIA listing has it. What you choose is the **image** —
`nvidia/cuda:*-devel-*` or any PyTorch image. `-runtime` images omit `nvcc`.

Three real listings, to make that concrete:

| | RTX 3070, $0.13/h | RTX A4000, $0.17/h | RTX 3090, $0.22/h |
|---|---|---|---|
| VRAM | 8 GB | 16 GB | 24 GB |
| System RAM | 35 GB | 31 GB | **60 GB** |
| vCPU | **18** | 12 | 16 |
| Verdict | Cheapest honest run | **Skip** — fewer cores and less RAM than both others | Best value per hour, see below |

The A4000 loses on every axis that matters and wins only on the one that does
not. Between the other two: the 3070 is the cheapest way to get one real data
point, and 8 GB produces the more interesting plan — a 30 B MoE leaves roughly
5.5 GiB for experts once hot tensors and an 8192-token KV cache are paid for, so
the planner must make real choices, where 16 GB keeps most of them and the
decision gets easy.

### A big card is a sweep rig, not a waste

Renting 24 GB looks like the mistake this section warns about, and it is — *if
you accept the plan it generates*, where the whole model fits and nothing is
offloaded. But `--reserve` holds VRAM back from the budget:

```bash
for mb in 16384 12288 8192 512; do
  of plan $MODEL --context 8192 --reserve $mb > plan-$mb.txt
done
```

On a 3090 those four runs produce the plans an 8 GB, 12 GB, 16 GB and 24 GB card
would get, and each can be benchmarked in the same session. Four calibration
points for one hour of rent is the fastest way to populate the registry this
project is ultimately for.

**What the simulation does not cover:** a withheld budget reproduces the
performance path — experts streaming over PCIe from host RAM while attention
runs on the GPU — but not VRAM exhaustion. It validates the cost model and the
`-ot` regex. It cannot tell you whether a real 8 GB card OOMs, because the
allocation never actually fails.

Ignore the 4090s and A100s at the top of every listing page. They are better
value per FLOP and, unless you are deliberately sweeping, useless here.

## RunPod, click by click

1. **Sign up** at [runpod.io](https://runpod.io) (Google or GitHub works), then
   **Billing → Add credit**. The minimum is around $10; it does not expire, and
   the hour itself costs a fraction of it.
2. *Optional, two minutes, worth it:* **Settings → SSH Public Keys**, paste
   `~/.ssh/id_ed25519.pub`. Without a key you are limited to the browser
   terminal, which cannot copy files out.       
3. **Pods → Deploy**, switch to **Community Cloud**, filter for the cards above.
4. **Read the card before clicking it.** Each listing shows its vCPU and RAM
   allocation. A 3060 attached to 8 vCPU and 31 GB of RAM is the target; the
   same GPU attached to 16 GB of RAM cannot hold the experts and will quietly
   benchmark your disk instead.
5. **Template: any `RunPod PyTorch 2.x` image.** Ubuntu with the CUDA toolkit
   preinstalled. Avoid anything described as *runtime* — those ship the CUDA
   libraries without `nvcc`, and `-DGGML_CUDA=ON` needs the compiler.
6. **Raise the disk before deploying**, to **60 GB**. The 20 GB default runs out
   during the model download, ten minutes in, after you have paid for the build.
7. **Deploy**, wait for *Running*, then **Connect → Start Web Terminal**, or
   `ssh root@POD_IP -p PORT`, substituting the two values it shows.
8. Run the steps below.
9. **Terminate** the pod — the trash icon, not *Stop*.

## Vast.ai, if you would rather pay less

Only the differences:

- **Register an SSH key first** (Account → SSH Keys). There is no browser
  terminal.
- Set **Disk Space to 60 GB with the slider before renting.** Disk is allocated
  at rent time and cannot be grown afterwards.
- Filter on **Reliability > 99%** and check **Inet Down**: you are pulling
  18 GB, and a 50 Mbit host turns that into 45 minutes of billed time.
- Take **On-Demand**, not **Interruptible** — the latter can be outbid and
  killed mid-benchmark.
- Image: `nvidia/cuda:12.4.1-devel-ubuntu22.04` or equivalent. **`-devel`, not
  `-runtime`**: same `nvcc` trap as above.
- Check **Max Duration** on the listing. Some hosts only guarantee availability
  for a few hours; anything over a day is plenty here.

## What it actually costs

| Item | Cost |
|---|---|
| GPU, 1.5 h at $0.20/h | ~$0.30 |
| Disk, 60 GB for two hours | a few cents |
| Egress | free on RunPod, occasionally billed on Vast.ai |
| **Minimum credit top-up** | **$5–10** |

So the experiment is €1 and the *entry ticket* is €5–10, which stays on the
account for next time. That is the honest number, and it is the reason step 0
exists: the highest-value finding in this document costs nothing to obtain.

## One caveat about calibrating on rented hardware

The CPU on a rented box is a slice of a large server shared with other tenants.
That does not touch the two pass/fail questions — the `-ot` command either
parses or it does not, and it either beats the baseline or it does not — but it
does undermine the derived `WEIGHT_READ_FACTOR`.

Two checks before trusting a bandwidth number:

```bash
of probe | grep -i bandwidth
sleep 120
of probe | grep -i bandwidth    # more than a few % apart → contended host
nproc                           # vs the vCPU count the listing advertised
```

If `nproc` reports 64 on a pod sold as 8 vCPU, the container is seeing the
host's cores: the all-threads bandwidth figure was measured with more threads
than you are entitled to, and it is optimistic.

Record it either way, and label it in the calibration file as measured on shared
hardware. A deviation from 1.25 found here is a reason to look, not a refutation.

## Steps

### 1. Launch, and prove CUDA works before anything else

**Run this first. Before the build, before the download, before anything.**

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

`True` and you may continue. `False` and you **destroy the instance and rent
another one** — do not debug it, do not build on it. This costs about one cent
and sixty seconds; finding out later costs an hour of rent and the whole run.

That is not hypothetical, and it is not rare. **Two rented pods, on two different
providers, failed identically**: a Vast.ai RTX 3070 and a RunPod RTX 5090. Both
passed every other check — `nvidia-smi` listed the card, the driver and the VRAM
— and both refused to initialise CUDA. PyTorch failed the same way on both, so
llama.cpp was never the problem.

Dropping to the driver API gives the real answer where PyTorch only says
"unknown":

```bash
python3 -c 'import ctypes; print(ctypes.CDLL("libcuda.so.1").cuInit(0))'
# 0 = fine.  999 = the failure described here.  803 = driver/library mismatch.
```

On both pods the cause was the same and visible in one line:

```bash
ls /dev/nvidia-caps/     # empty
```

Driver 580 initialises CUDA through the capability device nodes, and neither
container runtime created them. They cannot be added afterwards — `mknod` is not
permitted unprivileged — so the instance is unusable and the only move is to
destroy it. Check `/dev/nvidia-caps` and `cuInit` **before** anything else.

Kernel driver and userspace library matched on both (580.95.05), the device
nodes and permissions were correct, and the GPU was idle. None of the usual
suspects apply, which is exactly why the one-line check earns its place: no
amount of inspecting a broken pod makes it work.

Then the rest, before spending billed minutes on a machine that cannot do the job:

```bash
nvidia-smi        # the card, and its VRAM
free -g           # system RAM — but see below, this lies in a container
df -h /           # ≥ 60 GB free
nproc             # compare against the vCPU count the listing advertised
nvcc --version || ls /usr/local/cuda*/bin/nvcc
```

**`nvcc` is often present but off `PATH`.** Check `/usr/local/cuda/bin` before
concluding the toolkit is missing: `export PATH=/usr/local/cuda/bin:$PATH` costs
nothing, where redeploying costs fifteen minutes. Only if it genuinely is not
installed should you redeploy on a `-devel` or PyTorch template.

**`free -g` and `nproc` both report the host, not your container.** A pod
advertised as 35 GB and 18 vCPU reported 251 GB and 128 cores. The real figures
are in the cgroup:

```bash
cat /sys/fs/cgroup/memory.max     # bytes, or "max"
cat /sys/fs/cgroup/cpu.max        # "quota period", or "max"
```

`of probe` reads these for you and reports a *Usable* core count and a *Limit*
line when they apply. Trust those over `nproc`: benchmarking with 128 threads on
an 18-core entitlement measured 25.2 GB/s where 18 threads measured 74.5.

### 2. Build llama.cpp with CUDA

```bash
apt-get update && apt-get install -y cmake build-essential git tmux
tmux new -s work          # a closed browser tab kills the web terminal, not this
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build build --config Release -j$(nproc)
export PATH="$PWD/build/bin:$PATH"
llama-bench --help | head -5    # confirm it built
```

The build takes 5–10 minutes; it is most of the hour. If the compiler is killed
partway, `nproc` over-reported the cores you actually own — rebuild with `-j8`.

### 3. Install of-tune

```bash
cd ~
git clone https://github.com/RickyLeRoi/onfeather-tune
cd onfeather-tune
pip install -e .
```

### 4. Fetch a model that does not fit

`Qwen3-30B-A3B` at `Q4_K_M` is the canonical case: ~18 GB total, ~3 B active per
token, 128 experts of which 8 fire.

```bash
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf \
  --local-dir ~/models
export MODEL=~/models/Qwen3-30B-A3B-Q4_K_M.gguf
```

If that repository has moved, any MoE GGUF clearly larger than the card's VRAM
works. Record exactly which one you used.

### 5. Profile the machine

```bash
of probe -o ~/probe-cuda.json
of probe --json          # keep this output; it goes in the calibration file
```

**Check:** the GPU must be detected as vendor `nvidia` with the right VRAM, and
`unified_memory` must be false. If either is wrong, that is finding number one —
stop and report it.

### 6. Ask for a plan

```bash
of inspect $MODEL --context 8192
of plan $MODEL --profile ~/probe-cuda.json --context 8192
```

Record the printed `PROJECTED → Decode speed` and the full command. That number
is the prediction under test.

### 7. Measure the baseline — everything on CPU

```bash
llama-bench -m $MODEL -ngl 0 -p 512 -n 128 -o json > ~/bench-cpu.json
```

### 8. Measure the plan

Take the `-ot` argument from step 6 and hand it to `llama-bench`:

```bash
llama-bench -m $MODEL -ngl 99 \
  -ot "blk\.(N|N|...)\.ffn_(gate|up|down)_exps\.weight=CPU" \
  -p 512 -n 128 -o json > ~/bench-plan.json
```

**This is the moment of truth.** Three things can happen:

| Outcome | Meaning |
|---|---|
| Runs, `tg128` beats the CPU baseline | The path works. Record both numbers |
| Runs but no faster | The plan is wrong, not the mechanism. Record it — that is a cost-model bug worth having |
| **Errors out** | The generated command is malformed. **This is the most valuable possible result.** Capture the exact error |

### 9. Compare against `--n-cpu-moe`

The upstream coarse equivalent, for context on whether the finer plan earns its
existence:

```bash
llama-bench -m $MODEL -ngl 99 --n-cpu-moe 40 -p 512 -n 128 -o json > ~/bench-ncpumoe.json
```

### 10. Sanity-check the VRAM budget

While a benchmark runs, in a second shell:

```bash
watch -n1 nvidia-smi
```

Compare peak usage against the `VRAM BUDGET` block from step 6. Predicting 3.5
GiB and using 6 is a planner bug; predicting 3.5 and using 1.2 means the reserve
is far too conservative and speed is being left behind.

### 11. Write the calibration file

Follow the shape of `calibration/2026-07-25-i9-9880h.json`:

```bash
cd ~/onfeather-tune/calibration
cp 2026-07-25-i9-9880h.json $(date +%Y%m%d)-rtx3060.json
```

Fill in: the `of probe --json` output, llama.cpp commit, CUDA version, driver,
the exact model file, and for each run the command, `pp512`, `tg128` and tokens
generated. Include the failures — a plan that did not run is data.

### 12. Get the results off the box, then destroy it

```bash
tar czf /root/cuda-results.tar.gz \
  -C /root onfeather-tune/calibration bench-cpu.json bench-plan.json \
           bench-ncpumoe.json probe-cuda.json
```

With an SSH key registered, from your own machine:

```bash
PORT=12345 POD_IP=203.0.113.7        # both shown in RunPod's Connect panel
scp -P "$PORT" "root@$POD_IP:/root/cuda-results.tar.gz" ~/Downloads/
```

From a browser terminal there is no scp, so use RunPod's own peer-to-peer
transfer: `runpodctl send /root/cuda-results.tar.gz` prints a one-time code, and
`runpodctl receive CODE` on your laptop pulls the file down.

**Then terminate the pod** — *Terminate*, not *Stop*. A stopped pod still bills
for its disk, and one left running overnight costs more than the entire
experiment.

---

## What each result means

**The `-ot` command errors.** The highest-value outcome. Capture the message
verbatim: it means the regex or the buffer-type name is wrong, and every plan
the tool has ever printed is wrong with it.

**It runs and is faster.** The premise holds. Compute `measured tok/s × active
bytes per token` and compare against the STREAM figure from `of probe` — that
ratio is `WEIGHT_READ_FACTOR` on this hardware. If it is near 1.25, the constant
generalises. If not, it is machine-dependent and the community registry becomes
the point rather than a nicety.

**It runs and is not faster.** The mechanism works, the plan is bad. Try
different `--reserve` values and see whether the optimum sits somewhere the
planner is not looking.

**`--n-cpu-moe 40` beats the plan.** Also worth knowing, and worth saying out
loud in the README. The claim is that finer-grained placement wins; if the
coarse flag wins, the claim is wrong.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `failed to initialize CUDA: unknown error` / `cuInit -> 999` | Check `ls /dev/nvidia-caps/`. Empty means the runtime never created the capability nodes and the pod is unusable — destroy it. Seen on two providers |
| `nvcc: command not found` | Usually just `PATH` — try `/usr/local/cuda/bin/nvcc` first. Only redeploy if it is genuinely absent |
| `No space left on device` mid-download | Container disk left at its 20 GB default. Redeploy with 60 GB |
| Compiler killed partway through the build | `-j$(nproc)` with an over-reported core count. Rebuild with `-j8` |
| `of probe` reports no GPU | `nvidia-smi` missing from PATH inside the container |
| Two `of probe` runs disagree by 20% | Contended host. Fine for the pass/fail questions, unusable for calibration |
| Out of memory despite the plan | Reserve too low. Retry with `--reserve 1024` |
| `-ot` matches nothing | Shell ate the backslashes — single-quote the argument |
| Extremely slow prefill | CUDA build fell back to CPU. Check `cmake` output for `GGML_CUDA` |
| Decode barely beats the CPU baseline while the GPU sits idle | Experts spilled to disk: system RAM is below the model size |
