from onfeather_tune.model import (
    SCHEMA_VERSION,
    CpuInfo,
    GpuInfo,
    HardwareProfile,
    MemoryInfo,
    PlatformInfo,
)

GIB = 1024**3


def make_profile(**overrides) -> HardwareProfile:
    defaults = {
        "schema_version": SCHEMA_VERSION,
        "platform": PlatformInfo(system="Linux", release="6.1", machine="x86_64", python="3.11.0"),
        "cpu": CpuInfo(model="Ryzen 9 7950X", physical_cores=16, logical_cores=32, isa=["avx2"]),
        "memory": MemoryInfo(
            total_bytes=64 * GIB,
            available_bytes=48 * GIB,
            bandwidth_single_gbs=14.0,
            bandwidth_multi_gbs=61.0,
        ),
        "gpus": [GpuInfo(name="RTX 3050", vendor="nvidia", total_bytes=4 * GIB, free_bytes=3 * GIB)],
    }
    return HardwareProfile(**{**defaults, **overrides})


def test_fingerprint_is_stable_across_runs():
    assert make_profile().fingerprint() == make_profile().fingerprint()


def test_fingerprint_ignores_free_vram_and_available_ram():
    """Transient state must not fork the registry into one entry per run."""
    baseline = make_profile()
    busy = make_profile(
        memory=MemoryInfo(
            total_bytes=64 * GIB,
            available_bytes=9 * GIB,
            bandwidth_single_gbs=14.0,
            bandwidth_multi_gbs=61.0,
        ),
        gpus=[GpuInfo(name="RTX 3050", vendor="nvidia", total_bytes=4 * GIB, free_bytes=1 * GIB)],
    )
    assert baseline.fingerprint() == busy.fingerprint()


def test_fingerprint_tolerates_bandwidth_noise():
    """A few percent of benchmark jitter stays inside the same 5 GB/s bucket."""
    baseline = make_profile()
    jittered = make_profile(
        memory=MemoryInfo(
            total_bytes=64 * GIB,
            available_bytes=48 * GIB,
            bandwidth_single_gbs=13.6,
            bandwidth_multi_gbs=60.2,
        ),
    )
    assert baseline.fingerprint() == jittered.fingerprint()


def test_fingerprint_separates_different_gpus():
    other = make_profile(
        gpus=[GpuInfo(name="RTX 4090", vendor="nvidia", total_bytes=24 * GIB, free_bytes=23 * GIB)]
    )
    assert make_profile().fingerprint() != other.fingerprint()


def test_fingerprint_separates_single_from_dual_channel():
    """Same parts, half the bandwidth: a genuinely different machine to plan for."""
    crippled = make_profile(
        memory=MemoryInfo(
            total_bytes=64 * GIB,
            available_bytes=48 * GIB,
            bandwidth_single_gbs=12.0,
            bandwidth_multi_gbs=31.0,
        ),
    )
    assert make_profile().fingerprint() != crippled.fingerprint()


def test_fingerprint_survives_gpu_enumeration_order():
    igpu = GpuInfo(name="UHD 630", vendor="intel", total_bytes=1 * GIB, free_bytes=None)
    dgpu = GpuInfo(name="Radeon Pro 5500M", vendor="amd", total_bytes=4 * GIB, free_bytes=None)
    assert (
        make_profile(gpus=[igpu, dgpu]).fingerprint()
        == make_profile(gpus=[dgpu, igpu]).fingerprint()
    )


def test_schema_version_change_invalidates_fingerprint():
    assert make_profile().fingerprint() != make_profile(schema_version=99).fingerprint()


def test_to_dict_includes_fingerprint():
    data = make_profile().to_dict()
    assert data["fingerprint"] == make_profile().fingerprint()
    assert data["cpu"]["model"] == "Ryzen 9 7950X"
