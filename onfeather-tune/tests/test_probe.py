"""Hardware detection, with the container case first.

Rented GPUs live in containers, so a container is the normal environment for
this tool rather than an exotic one. Both bugs covered here were found by
running `of probe` on a rented pod: it reported the host's 251.5 GiB of RAM
against a real 32.6 GiB limit, and 128 cores against an entitlement of 18.
"""

from __future__ import annotations

from onfeather_tune import probe

GIB = 1024**3


# -- memory ---------------------------------------------------------------


def test_a_cgroup_limit_replaces_the_host_total(monkeypatch):
    """Planning against the host figure overstates what fits by 8x, and
    deciding what fits is the planner's entire job."""
    monkeypatch.setattr(probe, "_cgroup_memory_limit", lambda: 32 * GIB)
    info = probe.detect_memory(measure=False)

    assert info.total_bytes == 32 * GIB
    assert info.available_bytes <= 32 * GIB
    assert info.limit_bytes == 32 * GIB


def test_no_cgroup_limit_leaves_the_host_figures_alone(monkeypatch):
    monkeypatch.setattr(probe, "_cgroup_memory_limit", lambda: None)
    info = probe.detect_memory(measure=False)

    assert info.limit_bytes is None
    assert info.total_bytes > 0


def test_a_limit_larger_than_the_host_is_ignored(monkeypatch):
    """An unlimited cgroup writes a sentinel far above real RAM."""
    monkeypatch.setattr(probe, "_cgroup_memory_limit", lambda: 2**60)
    info = probe.detect_memory(measure=False)

    assert info.total_bytes < 2**60


def test_the_v1_unlimited_sentinel_reads_as_no_limit(tmp_path, monkeypatch):
    limit = tmp_path / "memory.limit_in_bytes"
    limit.write_text(str(2**63 - 4096))
    monkeypatch.setattr(probe, "Path", lambda _: limit)

    assert probe._cgroup_memory_limit() is None


def test_a_real_v2_limit_is_read(tmp_path, monkeypatch):
    limit = tmp_path / "memory.max"
    limit.write_text("34999996416\n")
    monkeypatch.setattr(probe, "Path", lambda _: limit)

    assert probe._cgroup_memory_limit() == 34999996416


# -- cpu ------------------------------------------------------------------


def test_a_cpu_quota_is_read_as_whole_cores(tmp_path, monkeypatch):
    quota = tmp_path / "cpu.max"
    quota.write_text("1800000 100000")
    monkeypatch.setattr(probe, "Path", lambda _: quota)

    assert probe._cgroup_cpu_quota() == 18


def test_an_unlimited_quota_reads_as_none(tmp_path, monkeypatch):
    quota = tmp_path / "cpu.max"
    quota.write_text("max 100000")
    monkeypatch.setattr(probe, "Path", lambda _: quota)

    assert probe._cgroup_cpu_quota() is None


def test_usable_cores_takes_the_quota_when_it_is_lower(monkeypatch):
    monkeypatch.setattr(probe, "_cgroup_cpu_quota", lambda: 18)
    assert probe.usable_cores() == 18


def test_usable_cores_survives_a_platform_without_affinity(monkeypatch):
    """macOS has no affinity mask; absence must not raise."""
    monkeypatch.setattr(probe, "_cgroup_cpu_quota", lambda: None)
    monkeypatch.delattr(probe.os, "sched_getaffinity", raising=False)

    assert probe.usable_cores() is None


def test_detect_cpu_reports_the_entitlement(monkeypatch):
    monkeypatch.setattr(probe, "usable_cores", lambda: 18)
    assert probe.detect_cpu().usable_cores == 18


# -- the two talking to each other ----------------------------------------


def test_bandwidth_is_never_measured_with_more_threads_than_allowed(monkeypatch):
    """128 threads on an 18-core share measured 25.2 GB/s where 18 measured 74.5.

    Benchmarking past the entitlement does not merely add noise, it reports a
    third of the real figure — and that figure is what predicts decode speed.
    """
    seen: list[int] = []
    monkeypatch.setattr(probe, "usable_cores", lambda: 4)
    monkeypatch.setattr(probe.psutil, "cpu_count", lambda logical=True: 64)
    monkeypatch.setattr(probe.bench, "measure_single_thread", lambda: 1.0)
    monkeypatch.setattr(probe.bench, "measure_multi_thread",
                        lambda count: seen.append(count) or 1.0)

    probe.detect_memory(measure=True)
    assert seen == [4]


def test_an_unrestricted_machine_still_benchmarks_on_physical_cores(monkeypatch):
    """The laptop calibration was taken at physical-core count; keep it valid."""
    seen: list[int] = []
    monkeypatch.setattr(probe, "usable_cores", lambda: 16)
    monkeypatch.setattr(probe.psutil, "cpu_count", lambda logical=True: 8)
    monkeypatch.setattr(probe.bench, "measure_single_thread", lambda: 1.0)
    monkeypatch.setattr(probe.bench, "measure_multi_thread",
                        lambda count: seen.append(count) or 1.0)

    probe.detect_memory(measure=True)
    assert seen == [8]
