"""Offload planning: decide where each tensor lives.

llama.cpp already ships the coarse version of this. `--n-cpu-moe N` pushes the
experts of N layers to CPU, and `--fit` auto-shrinks things to avoid an OOM. So
the job here is not "make it run" -- upstream does that -- it is **make it
fast**, in the space those flags cannot express:

  * per tensor *type*: `ffn_down_exps` is read differently from `ffn_up_exps`,
    and they need not share a fate;
  * per *layer*, with the VRAM left over after attention and the KV cache
    have been paid for;
  * against a *measured* bandwidth figure rather than a guess.

The plan is emitted as `-ot` regexes, which llama.cpp applies backend-agnostically.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from .estimate import DecodeEstimate, decode_estimate, kv_cache_bytes
from .gguf import HOT_ROLES, GGUFModel, TensorRole
from .model import GpuInfo, HardwareProfile

#: 20260725 RG VRAM beyond weights and KV cache: compute buffers, context, fragmentation.
DEFAULT_RESERVE_BYTES = 512 * 1024**2

#: 20260808 ** RG #Security Below this share of the card, someone else is holding the rest.
OCCUPIED_GPU_SHARE = 0.9

#: 20260725 RG Expert tensor types, in the order we give them up.
EXPERT_SURRENDER_ORDER = ("gate", "up", "down")


class PlanError(Exception):
    """Raised when no viable plan exists for the requested configuration."""


# 20260725 RG Below this share of RAM traffic the bandwidth model breaks: 1.45x at 26%.
RAM_BOUND_FLOOR = 0.30


@dataclass
class Plan:
    model: GGUFModel
    context_length: int

    gpu_bytes_available: int
    kv_cache_bytes: int
    hot_bytes: int
    expert_bytes_on_gpu: int
    expert_bytes_on_cpu: int

    cpu_expert_layers: list[int] = field(default_factory=list)
    """Layers whose routed experts are pushed to system RAM."""

    hot_on_cpu: bool = False
    """True when even the always-resident tensors did not fit, so everything runs
    on CPU. The GPU is then worth using only for prompt processing."""

    estimate: DecodeEstimate | None = None

    threads: int | None = None
    """Threads to run with. Emitted as `-t` because llama.cpp's default is the
    machine's core count, which inside a container is not the entitlement: on a
    128-core host allowing 18, decode measured 26.4 tok/s at 18 threads, 12.8 at
    36 and 6.7 at 64. Leaving it unset there costs a 4x slowdown silently."""

    # -- accounting -------------------------------------------------------

    @property
    def gpu_bytes_used(self) -> int:
        if self.hot_on_cpu:
            return 0
        return self.hot_bytes + self.expert_bytes_on_gpu + self.kv_cache_bytes

    @property
    def headroom_bytes(self) -> int:
        return self.gpu_bytes_available - self.gpu_bytes_used

    @property
    def ram_bound_fraction(self) -> float | None:
        """Share of the per-token weight traffic that crosses the memory bus.

        The cost model assumes that traffic is the bottleneck, which holds while
        it dominates and fails when it does not. Measured on a P100 against
        Qwen3-30B-A3B: at 100% the prediction was exact, at 26% it was 1.45x
        optimistic, and at 10% it was 2.16x optimistic — the GPU had become the
        limit and the model cannot see that.
        """
        total = self.model.active_bytes_per_token()
        return self.active_bytes_from_ram / total if total else None

    @property
    def active_bytes_from_ram(self) -> int:
        """Bytes crossing the memory bus per token.

        Only weights living in system RAM are counted: what sits in VRAM is read
        at GPU bandwidth, which is fast enough not to be the bottleneck.
        """
        if self.hot_on_cpu:
            return self.model.active_bytes_per_token()

        model = self.model
        if not model.is_moe or not model.expert_count:
            return self.expert_bytes_on_cpu

        fraction = min(model.expert_used_count / model.expert_count, 1.0)
        return int(self.expert_bytes_on_cpu * fraction)

    # -- rendering --------------------------------------------------------

    def override_tensor_patterns(self) -> list[str]:
        """The `-ot` arguments implementing this plan."""
        if self.hot_on_cpu:
            return [r"\.weight=CPU"]
        if not self.cpu_expert_layers:
            return []
        layers = _layers_pattern(self.cpu_expert_layers, self.model.block_count or 0)
        return [rf"blk\.{layers}\.ffn_(gate|up|down)_exps\.weight=CPU"]

    def llama_args(self, *, gpu_layers: int = 99) -> list[str]:
        """A complete llama.cpp argument list for this plan."""
        args: list[str] = ["-ngl", "0" if self.hot_on_cpu else str(gpu_layers)]
        for pattern in self.override_tensor_patterns():
            args += ["-ot", pattern]
        args += ["-c", str(self.context_length)]
        if self.threads:
            args += ["-t", str(self.threads)]
        return args

    def command(self, binary: str = "llama-server", model_path: str | None = None) -> str:
        path = model_path or (str(self.model.path) if self.model.path else "MODEL.gguf")
        parts = [binary, "-m", path, *self.llama_args()]
        return " ".join(_quote(part) for part in parts)


def _quote(part: str) -> str:
    return f'"{part}"' if any(character in part for character in " |\\()") else part


def _layers_pattern(layers: list[int], block_count: int) -> str:
    """Regex fragment matching exactly `layers`.

    Explicit alternation rather than compacted numeric ranges: a range regex is
    easy to get subtly wrong (`blk.1` matching `blk.10`), and a wrong pattern
    fails silently as a slow run rather than an error. The one shortcut taken is
    the whole-model case, where `\\d+` is unambiguous.
    """
    if not layers:
        raise PlanError("cannot build a pattern for an empty layer set")
    if block_count and len(layers) == block_count:
        return r"\d+"
    return "(" + "|".join(str(layer) for layer in sorted(layers)) + ")"


def make_plan(
    model: GGUFModel,
    profile: HardwareProfile,
    *,
    context_length: int | None = None,
    reserve_bytes: int = DEFAULT_RESERVE_BYTES,
    kv_bytes_per_element: int = 2,
) -> Plan:
    """Compute the fastest placement that fits.

    The strategy follows from the cost model rather than from taste: every hot
    tensor runs on every token, so a byte of VRAM spent on attention buys far
    more than a byte spent on experts. Hot tensors and the KV cache are paid
    first, and whatever remains is filled with routed experts.
    """
    context = context_length or model.context_length or 4096
    block_count = model.block_count or 0

    kv_bytes = 0
    if all((model.attention_layer_count, model.head_count_kv, model.head_dim)):
        kv_bytes = kv_cache_bytes(
            layer_count=model.attention_layer_count,
            head_count_kv=model.head_count_kv,
            head_dim=model.head_dim,
            context_length=context,
            bytes_per_element=kv_bytes_per_element,
        )

    available = _usable_vram(profile, reserve_bytes)
    hot_bytes = model.hot_bytes
    expert_bytes = model.routed_expert_bytes
    bandwidth = profile.memory.bandwidth_multi_gbs

    # 20260725 RG Nothing fits: run on CPU and stop pretending the GPU helps decoding.
    if available <= 0 or hot_bytes + kv_bytes > available:
        plan = Plan(
            model=model,
            context_length=context,
            gpu_bytes_available=max(available, 0),
            kv_cache_bytes=kv_bytes,
            hot_bytes=hot_bytes,
            expert_bytes_on_gpu=0,
            expert_bytes_on_cpu=expert_bytes,
            cpu_expert_layers=list(range(block_count)),
            hot_on_cpu=True,
        )
    else:
        remaining = available - hot_bytes - kv_bytes
        per_layer = expert_bytes / block_count if block_count else 0

        layers_on_gpu = int(remaining // per_layer) if per_layer > 0 else block_count
        layers_on_gpu = min(layers_on_gpu, block_count)
        cpu_layers = list(range(layers_on_gpu, block_count))

        plan = Plan(
            model=model,
            context_length=context,
            gpu_bytes_available=available,
            kv_cache_bytes=kv_bytes,
            hot_bytes=hot_bytes,
            expert_bytes_on_gpu=int(per_layer * layers_on_gpu),
            expert_bytes_on_cpu=int(per_layer * len(cpu_layers)),
            cpu_expert_layers=cpu_layers,
        )

    plan.threads = _recommended_threads(profile)
    if bandwidth:
        plan.estimate = decode_estimate(plan.active_bytes_from_ram, bandwidth)
    return plan


def _recommended_threads(profile: HardwareProfile) -> int | None:
    """Threads to pass as `-t`, or None to leave llama.cpp's default alone.

    Only emitted when the entitlement is below the machine's physical core
    count — that is, in a container. On ordinary hardware llama.cpp's own
    default is fine and an explicit `-t` would just be noise in the command.
    """
    cpu = profile.cpu
    allowed, physical = cpu.usable_cores, cpu.physical_cores
    if not allowed or not physical or allowed >= physical:
        return None
    return allowed


def _capacity(gpu: GpuInfo) -> int:
    """VRAM to plan against. Zero free is a measurement, not a missing value.

    `free_bytes or total_bytes` read a fully occupied GPU as an unmeasured one
    and planned against the whole card, which is an OOM every time on the
    shared, rented hardware this tool is aimed at.
    """
    if gpu.free_bytes is not None:
        return gpu.free_bytes
    return gpu.total_bytes or 0


def _usable_vram(profile: HardwareProfile, reserve_bytes: int) -> int:
    """VRAM we are willing to fill on the largest usable GPU.

    Unified-memory GPUs are skipped: their 'VRAM' is the same DRAM the CPU
    already reads, so moving a tensor there buys no bandwidth and the whole
    hot/cold split stops meaning anything.
    """
    candidates = [gpu for gpu in profile.gpus if not gpu.unified_memory and gpu.total_bytes]
    if not candidates:
        return 0

    best = max(candidates, key=_capacity)
    capacity = _capacity(best)

    if best.total_bytes and capacity < best.total_bytes * OCCUPIED_GPU_SHARE:
        # 20260808 ** RG #Security Something else holds the card; the plan lasts only while it does.
        print(
            f"note: planning against {capacity / 1024**3:.2f} GiB free of "
            f"{best.total_bytes / 1024**3:.2f} GiB on {best.name} — something else "
            f"holds the rest, and this plan is only valid while that stays true",
            file=sys.stderr,
        )
    return capacity - reserve_bytes


def summarise(plan: Plan) -> str:
    """Human-readable plan, including the command to run it."""
    lines: list[str] = []

    def row(label: str, value: object) -> None:
        lines.append(f"  {label:<22} {value}")

    lines.append("PLAN")
    if plan.hot_on_cpu:
        row("Placement", "everything on CPU (GPU too small for the hot set)")
    else:
        resident = (plan.model.block_count or 0) - len(plan.cpu_expert_layers)
        row("Hot tensors", "GPU")
        row("Expert layers on GPU", f"{resident} of {plan.model.block_count}")
        row("Expert layers on CPU", f"{len(plan.cpu_expert_layers)} of {plan.model.block_count}")

    lines.append("")
    lines.append("VRAM BUDGET")
    row("Available", _gib(plan.gpu_bytes_available))
    row("Hot tensors", _gib(plan.hot_bytes if not plan.hot_on_cpu else 0))
    row("KV cache", f"{_gib(plan.kv_cache_bytes)} @ {plan.context_length}")
    row("Experts kept", _gib(plan.expert_bytes_on_gpu))
    row("Headroom", _gib(plan.headroom_bytes))

    if plan.estimate:
        lines.append("")
        lines.append("PROJECTED")
        row("Read from RAM/token", _gib(plan.active_bytes_from_ram))
        row("Decode speed", plan.estimate.describe())
        if plan.ram_bound_fraction is not None and plan.ram_bound_fraction < RAM_BOUND_FLOOR:
            lines.append(
                f"  {'':<22} only {plan.ram_bound_fraction:.0%} of the active weights\n"
                f"  {'':<22} cross the memory bus, so this figure is optimistic —\n"
                f"  {'':<22} the GPU, not RAM, is the limit here. Measured 2.2x\n"
                f"  {'':<22} slower than predicted at 10% on a P100."
            )

    lines.append("")
    lines.append("COMMAND")
    lines.append(f"  {plan.command()}")
    return "\n".join(lines)


def _gib(value: int) -> str:
    return f"{value / 1024**3:.2f} GiB"


# 20260725 RG Re-exported so callers reason about placement without importing gguf.
__all__ = ["Plan", "PlanError", "make_plan", "summarise", "HOT_ROLES", "TensorRole"]
