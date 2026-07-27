"""Command line entry point for `of`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, estimate, gguf
from . import plan as plan_module
from . import probe as probe_module
from .model import CpuInfo, GpuInfo, HardwareProfile, MemoryInfo, PlatformInfo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="of",
        description="OnFeather — run heavy models on light hardware.",
    )
    parser.add_argument("--version", action="version", version=f"onfeather-tune {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    probe_parser = subcommands.add_parser(
        "probe", help="inspect this machine and emit a hardware profile"
    )
    probe_parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    probe_parser.add_argument("-o", "--out", type=Path, help="also write the profile to a file")
    probe_parser.add_argument(
        "--no-bandwidth",
        action="store_true",
        help="skip the memory bandwidth benchmark (faster, but the planner loses its "
        "best predictor of CPU decode speed)",
    )
    probe_parser.set_defaults(handler=_cmd_probe)

    inspect_parser = subcommands.add_parser(
        "inspect", help="break down a GGUF model and estimate its speed here"
    )
    inspect_parser.add_argument("model", type=Path, help="path to a .gguf file")
    inspect_parser.add_argument(
        "--bandwidth",
        type=float,
        metavar="GBS",
        help="memory bandwidth in GB/s (default: measure this machine)",
    )
    inspect_parser.add_argument(
        "--context", type=int, help="context length for the KV cache estimate (default: model max)"
    )
    inspect_parser.set_defaults(handler=_cmd_inspect)

    plan_parser = subcommands.add_parser(
        "plan", help="compute the best offload split for this machine"
    )
    plan_parser.add_argument("model", type=Path, help="path to a .gguf file")
    plan_parser.add_argument(
        "--context", type=int, help="context length to plan for (default: model max)"
    )
    plan_parser.add_argument(
        "--reserve",
        type=int,
        metavar="MB",
        default=plan_module.DEFAULT_RESERVE_BYTES // 1024**2,
        help="VRAM held back for compute buffers (default: %(default)s MB)",
    )
    plan_parser.add_argument(
        "--profile", type=Path, help="hardware profile JSON from `of probe` (default: probe now)"
    )
    plan_parser.set_defaults(handler=_cmd_plan)

    args = parser.parse_args(argv)
    return args.handler(args)


def _cmd_probe(args: argparse.Namespace) -> int:
    measure = not args.no_bandwidth
    if measure and not args.json:
        print("Measuring memory bandwidth (a few seconds)...", file=sys.stderr)

    profile = probe_module.probe(measure_bandwidth=measure)

    if args.json:
        print(profile.to_json())
    else:
        print(_render(profile))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(profile.to_json() + "\n")
        print(f"\nProfile written to {args.out}", file=sys.stderr)

    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    if not args.model.exists():
        print(f"error: {args.model} does not exist", file=sys.stderr)
        return 1

    try:
        model = gguf.read(args.model)
    except gguf.GGUFError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    bandwidth = args.bandwidth
    if bandwidth is None:
        print("Measuring memory bandwidth (a few seconds)...", file=sys.stderr)
        bandwidth = probe_module.detect_memory().bandwidth_multi_gbs

    print(_render_model(model, bandwidth, args.context))
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    if not args.model.exists():
        print(f"error: {args.model} does not exist", file=sys.stderr)
        return 1

    try:
        model = gguf.read(args.model)
    except gguf.GGUFError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.profile:
        profile = _load_profile(args.profile)
    else:
        print("Probing hardware (a few seconds)...", file=sys.stderr)
        profile = probe_module.probe()

    plan = plan_module.make_plan(
        model,
        profile,
        context_length=args.context,
        reserve_bytes=args.reserve * 1024**2,
    )
    print(plan_module.summarise(plan))
    return 0


def _load_profile(path: Path) -> HardwareProfile:
    """Rebuild a profile saved by `of probe -o`."""
    data = json.loads(path.read_text())
    return HardwareProfile(
        schema_version=data["schema_version"],
        platform=PlatformInfo(**data["platform"]),
        cpu=CpuInfo(**data["cpu"]),
        memory=MemoryInfo(**data["memory"]),
        gpus=[GpuInfo(**gpu) for gpu in data["gpus"]],
    )


def _render_model(model: gguf.GGUFModel, bandwidth: float | None, context: int | None) -> str:
    lines: list[str] = []

    def row(label: str, value: object) -> None:
        lines.append(f"  {label:<22} {value}")

    lines.append("MODEL")
    row("Architecture", model.architecture)
    row("Layers", model.block_count)
    row("Total weights", _gib(model.total_bytes))
    if model.is_moe:
        row("Experts", f"{model.expert_used_count} active of {model.expert_count} per token")
    else:
        row("Type", "dense")

    lines.append("")
    lines.append("WEIGHT BREAKDOWN")
    totals = model.bytes_by_role()
    for role, size in sorted(totals.items(), key=lambda item: -item[1]):
        share = size / model.total_bytes * 100 if model.total_bytes else 0
        marker = "hot " if role in gguf.HOT_ROLES else "cold"
        row(f"[{marker}] {role.name.lower()}", f"{_gib(size):>10}  {share:5.1f}%")

    lines.append("")
    lines.append("PER TOKEN")
    active = model.active_bytes_per_token()
    row("Always resident", _gib(model.hot_bytes))
    row("Offloadable", _gib(model.cold_bytes))
    row("Read per token", f"{_gib(active)}  ({active / model.total_bytes:.0%} of the model)")

    context_length = context or _default_context(model)
    if all((model.attention_layer_count, model.head_count_kv, model.head_dim, context_length)):
        kv_bytes = estimate.kv_cache_bytes(
            layer_count=model.attention_layer_count,
            head_count_kv=model.head_count_kv,
            head_dim=model.head_dim,
            context_length=context_length,
        )
        suffix = ""
        if model.context_length and context_length < model.context_length:
            suffix = f", model max {model.context_length}"
        row(f"KV cache @ {context_length}", f"{_gib(kv_bytes)} (f16{suffix})")

    if bandwidth:
        result = estimate.decode_estimate(active, bandwidth)
        lines.append("")
        lines.append("ESTIMATE (all weights in system RAM)")
        row("Memory bandwidth", f"{bandwidth:.1f} GB/s")
        row("Decode speed", result.describe())
        lines.append("")
        lines.append(
            f"  Effective read bandwidth {result.effective_bandwidth_gbs:.1f} GB/s "
            f"({estimate.WEIGHT_READ_FACTOR:.2f}x the STREAM\n"
            "  figure: weight streaming is pure sequential read). Calibrated on one\n"
            "  machine — see calibration/. Offloading hot tensors to GPU beats this."
        )

    return "\n".join(lines)


def _render(profile: HardwareProfile) -> str:
    lines: list[str] = []

    def row(label: str, value: object) -> None:
        lines.append(f"  {label:<20} {value}")

    lines.append("PLATFORM")
    row("System", f"{profile.platform.system} {profile.platform.release}")
    row("Arch", profile.platform.machine)

    lines.append("")
    lines.append("CPU")
    row("Model", profile.cpu.model)
    row("Cores", f"{profile.cpu.physical_cores} physical / {profile.cpu.logical_cores} logical")
    allowed = profile.cpu.usable_cores
    if allowed and profile.cpu.physical_cores and allowed < profile.cpu.physical_cores:
        # 20260726 ** RG Only worth saying when it contradicts the line above.
        row("Usable", f"{allowed}  (restricted — this is a container)")
    row("ISA", ", ".join(profile.cpu.isa) or "not detected")

    memory = profile.memory
    lines.append("")
    lines.append("MEMORY")
    row("Total", _gib(memory.total_bytes))
    row("Available", _gib(memory.available_bytes))
    if memory.limit_bytes:
        row("Limit", f"{_gib(memory.limit_bytes)}  (cgroup — the host has more)")
    if memory.bandwidth_single_gbs is not None:
        row("Bandwidth (1 thread)", f"{memory.bandwidth_single_gbs:.1f} GB/s")
    if memory.bandwidth_multi_gbs is not None:
        threads = f" @ {allowed} threads" if allowed else ""
        row("Bandwidth (all)", f"{memory.bandwidth_multi_gbs:.1f} GB/s{threads}")

    lines.append("")
    lines.append("GPU")
    if not profile.gpus:
        row("", "none detected — CPU-only planning")
    for gpu in profile.gpus:
        row("Name", f"{gpu.name} ({gpu.vendor})")
        row("Memory", _gib(gpu.total_bytes))
        if gpu.free_bytes is not None:
            row("Free", _gib(gpu.free_bytes))
        if gpu.unified_memory:
            row("Type", "unified with system RAM")
        if gpu.driver:
            row("Driver", gpu.driver)

    lines.append("")
    lines.append(f"FINGERPRINT  {profile.fingerprint()}")
    return "\n".join(lines)


#: 20260726 ** RG Context used for the KV estimate when the caller does not say.
DEFAULT_CONTEXT = 8192


def _default_context(model: gguf.GGUFModel) -> int:
    return min(model.context_length or DEFAULT_CONTEXT, DEFAULT_CONTEXT)


def _gib(value: int | None) -> str:
    return "unknown" if value is None else f"{value / 1024**3:.1f} GiB"


if __name__ == "__main__":
    raise SystemExit(main())
