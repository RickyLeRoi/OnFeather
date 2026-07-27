"""Schema of a hardware profile.

The profile is the input to the planner and the key of the community config
registry, so its shape is versioned: a change to SCHEMA_VERSION invalidates
previously published plans.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

SCHEMA_VERSION = 1


@dataclass
class CpuInfo:
    model: str
    physical_cores: int | None
    logical_cores: int | None
    isa: list[str] = field(default_factory=list)
    """ISA extensions that change llama.cpp's kernel selection (avx2, avx512f, amx_bf16, neon...)."""
    usable_cores: int | None = None
    """Cores this process may actually use, which in a container is not the
    machine's count. On a rented 128-core host entitled to 18, decode measured
    26.4 tok/s at 18 threads and 6.7 at 64 — so the difference is not cosmetic."""


@dataclass
class MemoryInfo:
    total_bytes: int
    available_bytes: int
    bandwidth_single_gbs: float | None = None
    bandwidth_multi_gbs: float | None = None
    """Aggregate multi-threaded bandwidth. This is the number that predicts
    decode speed for experts held in RAM, so it matters more than core count."""
    limit_bytes: int | None = None
    """A cgroup memory limit, when one applies. `total_bytes` is already clamped
    to it: a rented container reported 251.5 GiB of host RAM against a real
    limit of 32.6 GiB, and planning against the larger figure overstates by 8x."""


@dataclass
class GpuInfo:
    name: str
    vendor: str
    """One of: nvidia, amd, apple, intel, unknown."""
    total_bytes: int | None
    free_bytes: int | None
    unified_memory: bool = False
    """True on Apple Silicon and iGPUs, where 'VRAM' is carved out of system RAM
    and the offload problem is a different one."""
    driver: str | None = None


@dataclass
class PlatformInfo:
    system: str
    release: str
    machine: str
    python: str


@dataclass
class HardwareProfile:
    schema_version: int
    platform: PlatformInfo
    cpu: CpuInfo
    memory: MemoryInfo
    gpus: list[GpuInfo]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["fingerprint"] = self.fingerprint()
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def fingerprint(self) -> str:
        """Stable ID for 'a machine like this one'.

        Values are bucketed so that run-to-run noise (free VRAM, a benchmark
        landing a few percent off) does not fork the registry into one entry
        per run, while a genuinely different machine still gets its own key.
        """
        gpu_parts = [
            f"{g.vendor}:{g.name}:{_bucket_bytes(g.total_bytes)}"
            for g in sorted(self.gpus, key=lambda g: g.name)
        ]
        parts = [
            f"v{self.schema_version}",
            self.platform.system,
            self.platform.machine,
            self.cpu.model,
            str(self.cpu.physical_cores),
            "+".join(sorted(self.cpu.isa)),
            _bucket_bytes(self.memory.total_bytes),
            _bucket_bandwidth(self.memory.bandwidth_multi_gbs),
            "|".join(gpu_parts) or "none",
        ]
        digest = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()
        return digest[:16]


def _bucket_bytes(value: int | None) -> str:
    """Round to the nearest GiB. Installed RAM and VRAM are reported with small
    variations depending on the source; the GiB is the meaningful unit."""
    if value is None:
        return "na"
    return f"{round(value / 1024**3)}Gi"


def _bucket_bandwidth(value: float | None) -> str:
    """Round to 5 GB/s. Finer resolution than that is measurement noise."""
    if value is None:
        return "na"
    return f"{round(value / 5) * 5}"
