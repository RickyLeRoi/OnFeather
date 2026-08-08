"""Memory bandwidth microbenchmarks.

CPU-side inference of an offloaded MoE is bandwidth-bound, not FLOP-bound: every
token drags the active experts across the memory bus. Measured bandwidth is
therefore the single best predictor of decode speed, and it is what the planner
uses to decide whether moving a tensor to CPU costs 2 tok/s or 20.

Nominal specs are not a substitute for measuring: the same DDR5-5600 kit runs at
half its rated bandwidth in single-channel, and that halving is invisible to
every source except a real measurement.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# 20260725 RG Sized to overflow any realistic L3, so we measure DRAM not cache.
DEFAULT_ARRAY_MB = 256
DEFAULT_ITERATIONS = 5

# 20260808 ** RG #Security Footprint ceiling over every thread: 128 of them asked for 12 GB.
MAX_TOTAL_MB = 2048

# 20260725 RG Below this a thread's share fits in cache and measures the wrong thing.
MIN_PER_THREAD_MB = 32

#: 20260725 RG _allocate builds three arrays, so a workspace costs three times its size.
ARRAYS_PER_WORKSPACE = 3


def _kernel(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
    """STREAM 'add': a = b + c. Two reads and one write per element.

    Deliberately not STREAM 'triad' (a = b + scalar * c): NumPy cannot fuse the
    scalar multiply, so expressing triad takes two ufunc calls and moves five
    array-passes of traffic rather than three. Counting it as three understates
    bandwidth by ~1.7x. 'add' is a single ufunc call whose traffic is exactly
    the three passes we bill it for.
    """
    np.add(b, c, out=a)


def _time_kernel(arrays: tuple[np.ndarray, np.ndarray, np.ndarray], iterations: int) -> float:
    """Return the best wall-clock time over `iterations` runs.

    Best-of rather than mean: we want the machine's capability, and every source
    of noise (scheduler preemption, a background process) can only slow a run
    down, never speed it up.
    """
    a, b, c = arrays
    _kernel(a, b, c)
    best = float("inf")
    for _ in range(iterations):
        start = time.perf_counter()
        _kernel(a, b, c)
        best = min(best, time.perf_counter() - start)
    return best


def _allocate(size_mb: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = (size_mb * 1024 * 1024) // np.dtype(np.float32).itemsize
    return (
        np.zeros(n, dtype=np.float32),
        np.ones(n, dtype=np.float32),
        np.full(n, 2.0, dtype=np.float32),
    )


def _bytes_moved(size_mb: int) -> int:
    """The kernel touches three arrays: two read, one written."""
    return size_mb * 1024 * 1024 * 3


def measure_single_thread(
    size_mb: int = DEFAULT_ARRAY_MB, iterations: int = DEFAULT_ITERATIONS
) -> float:
    """Bandwidth in GB/s available to one core."""
    arrays = _allocate(size_mb)
    seconds = _time_kernel(arrays, iterations)
    return _bytes_moved(size_mb) / seconds / 1e9


def workspace_plan(threads: int, size_mb: int = DEFAULT_ARRAY_MB) -> tuple[int, int]:
    """How many threads to run and how much each gets, inside the ceiling.

    Ordinarily the total is held constant by splitting `size_mb` between the
    threads. The per-thread floor is what breaks that: below it a thread's
    share fits in cache and measures the wrong thing, so past a certain count
    the total starts growing again — 128 threads at the floor is 12 GB of
    workspace for a microbenchmark, which is a MemoryError or the OOM killer
    on the exact machines the floor exists for.

    So the footprint wins over the thread count. Bandwidth saturates well
    before a hundred threads on any real bus, which makes measuring fewer of
    them a far smaller error than not measuring at all.
    """
    per_thread_mb = max(size_mb // threads, MIN_PER_THREAD_MB)
    budget_mb = MAX_TOTAL_MB // ARRAYS_PER_WORKSPACE

    if per_thread_mb * threads > budget_mb:
        threads = max(1, budget_mb // MIN_PER_THREAD_MB)
        per_thread_mb = MIN_PER_THREAD_MB
    return threads, per_thread_mb


def measure_multi_thread(
    threads: int,
    size_mb: int = DEFAULT_ARRAY_MB,
    iterations: int = DEFAULT_ITERATIONS,
) -> float:
    """Aggregate bandwidth in GB/s with `threads` cores hammering the bus.

    NumPy releases the GIL inside these ufunc loops, so threads (rather than
    processes) genuinely run in parallel here and we avoid paying to copy
    hundreds of megabytes into worker processes.

    Each thread gets its own arrays: sharing them would let the caches serve
    part of the traffic and inflate the result. That makes the footprint grow
    with the thread count, so it is capped — see `workspace_plan`, and note
    that past the ceiling this measures fewer threads than it was asked for.
    """
    if threads < 1:
        raise ValueError("threads must be >= 1")

    threads, per_thread_mb = workspace_plan(threads, size_mb)
    workspaces = [_allocate(per_thread_mb) for _ in range(threads)]

    # 20260725 RG Warm every workspace so page faults land outside the timing.
    for arrays in workspaces:
        _kernel(*arrays)

    best = float("inf")
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for _ in range(iterations):
            start = time.perf_counter()
            list(pool.map(lambda arrays: _kernel(*arrays), workspaces))
            best = min(best, time.perf_counter() - start)

    return _bytes_moved(per_thread_mb) * threads / best / 1e9
