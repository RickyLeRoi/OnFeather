"""Hardware detection.

Everything here is best-effort and degrades to None rather than raising: a
missing vendor tool must not stop a probe, because a partial profile is still
enough to plan with.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

import psutil

from . import bench
from .model import SCHEMA_VERSION, CpuInfo, GpuInfo, HardwareProfile, MemoryInfo, PlatformInfo

# 20260726 ** RG ISA extensions llama.cpp actually branches on when picking kernels.
INTERESTING_ISA = {
    "avx", "avx2", "avx512f", "avx512bw", "avx512vl", "avx512dq", "avx512_vnni",
    "avx512_bf16", "amx_tile", "amx_int8", "amx_bf16", "f16c", "fma", "neon",
    "asimd", "dotprod", "i8mm", "sve", "sve2",
}


def _run(command: list[str], timeout: float = 10.0) -> str | None:
    """Run a command, returning stdout or None if it is unavailable or fails."""
    if shutil.which(command[0]) is None:
        return None
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return result.stdout if result.returncode == 0 else None


def detect_platform() -> PlatformInfo:
    return PlatformInfo(
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        python=platform.python_version(),
    )


def detect_cpu() -> CpuInfo:
    return CpuInfo(
        model=_cpu_model(),
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(logical=True),
        isa=sorted(_cpu_isa()),
        usable_cores=usable_cores(),
    )


def usable_cores() -> int | None:
    """How many cores this process may actually use.

    `cpu_count` reports the machine, and a container is routinely a slice of a
    much larger one. Rented GPUs live in containers, so this is the normal case
    rather than the exotic one.
    """
    limits = []
    # 20260726 ** RG macOS and Windows expose no affinity mask.
    with contextlib.suppress(AttributeError, OSError):
        limits.append(len(os.sched_getaffinity(0)))
    quota = _cgroup_cpu_quota()
    if quota:
        limits.append(quota)
    return min(limits) if limits else None


def _cgroup_cpu_quota() -> int | None:
    """Whole cores allowed by a CFS quota, if one is set."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError, IndexError):
        pass
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    return None


def _cgroup_memory_limit() -> int | None:
    """A container's memory ceiling, if it has one lower than the host's."""
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        # 20260726 ** RG cgroup v1 writes a sentinel near 2^63 to mean unlimited.
        if 0 < value < 2**62:
            return value
    return None


def _cpu_model() -> str:
    system = platform.system()
    if system == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out.strip()
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _cpu_isa() -> set[str]:
    system = platform.system()
    flags: set[str] = set()

    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as handle:
                for line in handle:
                    if line.startswith(("flags", "Features")):
                        flags.update(line.split(":", 1)[1].split())
                        break
        except OSError:
            pass

    elif system == "Darwin":
        # 20260726 ** RG Intel Macs expose one feature string; Apple Silicon one sysctl each.
        for key in ("machdep.cpu.features", "machdep.cpu.leaf7_features"):
            out = _run(["sysctl", "-n", key])
            if out:
                flags.update(token.lower().replace(".", "_") for token in out.split())
        out = _run(["sysctl", "-a"])
        if out:
            for line in out.splitlines():
                match = re.match(r"hw\.optional\.(?:arm\.)?(\w+):\s*1$", line.strip())
                if match:
                    flags.add(match.group(1).lower().replace("feat_", ""))

    elif system == "Windows":
        # 20260726 ** RG No cheap flag dump on Windows; the planner assumes the baseline.
        pass

    return {flag for flag in flags if flag in INTERESTING_ISA}


def detect_memory(*, measure: bool = True, threads: int | None = None) -> MemoryInfo:
    virtual = psutil.virtual_memory()
    total, available = virtual.total, virtual.available

    limit = _cgroup_memory_limit()
    if limit and limit < total:
        # 20260726 ** RG Report what this process can have, not what the host owns.
        total, available = limit, min(available, limit)

    info = MemoryInfo(total_bytes=total, available_bytes=available, limit_bytes=limit)
    if measure:
        physical = psutil.cpu_count(logical=False) or 1
        allowed = usable_cores()
        # 20260726 ** RG Never benchmark with more threads than we are entitled to:
        # 128 threads on an 18-core share measured 25 GB/s where 18 measured 74.5.
        worker_count = threads or (min(physical, allowed) if allowed else physical)
        info.bandwidth_single_gbs = round(bench.measure_single_thread(), 2)
        info.bandwidth_multi_gbs = round(bench.measure_multi_thread(worker_count), 2)
    return info


def detect_gpus() -> list[GpuInfo]:
    return _nvidia_gpus() or _amd_gpus() or _apple_gpus() or []


def _nvidia_gpus() -> list[GpuInfo]:
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return []

    gpus = []
    for line in out.strip().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        name, total_mib, free_mib, driver = fields
        gpus.append(
            GpuInfo(
                name=name,
                vendor="nvidia",
                total_bytes=_mib_to_bytes(total_mib),
                free_bytes=_mib_to_bytes(free_mib),
                driver=driver,
            )
        )
    return gpus


def _amd_gpus() -> list[GpuInfo]:
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--showproductname", "--json"])
    if not out:
        return []
    try:
        payload = json.loads(out)
    except json.JSONDecodeError:
        return []

    gpus = []
    for card, values in payload.items():
        if not card.lower().startswith("card") or not isinstance(values, dict):
            continue
        total = _first_int(values, "vram total memory")
        used = _first_int(values, "vram total used memory")
        gpus.append(
            GpuInfo(
                name=_first_str(values, "card series", "card model") or card,
                vendor="amd",
                total_bytes=total,
                free_bytes=(total - used) if total is not None and used is not None else None,
            )
        )
    return gpus


def _apple_gpus() -> list[GpuInfo]:
    if platform.system() != "Darwin":
        return []
    out = _run(["system_profiler", "-json", "SPDisplaysDataType"], timeout=30.0)
    if not out:
        return []
    try:
        entries = json.loads(out).get("SPDisplaysDataType", [])
    except json.JSONDecodeError:
        return []

    total_ram = psutil.virtual_memory().total
    gpus = []
    for entry in entries:
        name = entry.get("sppci_model", "Apple GPU")
        dedicated = entry.get("spdisplays_vram")
        shared = entry.get("spdisplays_vram_shared")
        # 20260726 ** RG Apple Silicon shares VRAM: the GPU addresses system RAM directly.
        if dedicated:
            total_bytes, unified = _parse_apple_vram(dedicated), False
        elif shared:
            total_bytes, unified = _parse_apple_vram(shared), True
        else:
            total_bytes, unified = total_ram, True
        gpus.append(
            GpuInfo(
                name=name,
                vendor=_apple_gpu_vendor(name),
                total_bytes=total_bytes,
                free_bytes=None,
                unified_memory=unified,
            )
        )
    return gpus


def _apple_gpu_vendor(name: str) -> str:
    """Identify the silicon behind a GPU listed by system_profiler.

    Intel Macs list a discrete AMD GPU alongside the Intel iGPU, so 'runs on
    macOS' says nothing about the vendor -- and the vendor decides which
    llama.cpp backend applies (Metal everywhere, but only Apple Silicon gets
    unified memory worth planning around).
    """
    lowered = name.lower()
    if "apple" in lowered:
        return "apple"
    if "amd" in lowered or "radeon" in lowered:
        return "amd"
    if "intel" in lowered:
        return "intel"
    if "nvidia" in lowered or "geforce" in lowered or "quadro" in lowered:
        return "nvidia"
    return "unknown"


def _parse_apple_vram(value: str | int) -> int | None:
    if isinstance(value, int):
        return value * 1024**2
    match = re.match(r"([\d.]+)\s*(MB|GB)", str(value), re.IGNORECASE)
    if not match:
        return None
    amount = float(match.group(1))
    return int(amount * (1024**3 if match.group(2).upper() == "GB" else 1024**2))


def _mib_to_bytes(value: str) -> int | None:
    try:
        return int(float(value) * 1024**2)
    except ValueError:
        return None


def _first_int(values: dict, *keys: str) -> int | None:
    for key, value in values.items():
        if key.lower() in keys:
            try:
                return int(str(value), 0)
            except ValueError:
                return None
    return None


def _first_str(values: dict, *keys: str) -> str | None:
    for wanted in keys:
        for key, value in values.items():
            if key.lower() == wanted and str(value).strip():
                return str(value).strip()
    return None


def probe(*, measure_bandwidth: bool = True) -> HardwareProfile:
    """Build a full hardware profile for this machine."""
    return HardwareProfile(
        schema_version=SCHEMA_VERSION,
        platform=detect_platform(),
        cpu=detect_cpu(),
        memory=detect_memory(measure=measure_bandwidth),
        gpus=detect_gpus(),
    )
