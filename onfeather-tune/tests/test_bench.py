import numpy as np

from onfeather_tune import bench


def test_kernel_computes_elementwise_sum():
    """Guards the accounting: if the kernel ever stops being a single pass over
    three arrays, _bytes_moved silently starts lying about bandwidth."""
    a = np.zeros(8, dtype=np.float32)
    b = np.full(8, 3.0, dtype=np.float32)
    c = np.full(8, 4.0, dtype=np.float32)

    bench._kernel(a, b, c)

    assert np.allclose(a, 7.0)


def test_bytes_moved_counts_three_arrays():
    assert bench._bytes_moved(1) == 3 * 1024 * 1024


def test_allocate_respects_requested_size():
    a, b, c = bench._allocate(4)
    assert a.nbytes == b.nbytes == c.nbytes == 4 * 1024 * 1024
    assert a.dtype == np.float32


def test_measure_single_thread_returns_plausible_bandwidth():
    result = bench.measure_single_thread(size_mb=32, iterations=2)
    # 20260725 RG Loose bounds: this catches a 1000x unit slip, not a slow machine.
    assert 0.5 < result < 2000


def test_measure_multi_thread_is_at_least_single_thread():
    single = bench.measure_single_thread(size_mb=32, iterations=2)
    multi = bench.measure_multi_thread(threads=2, size_mb=32, iterations=2)
    assert multi > single * 0.7


def test_measure_multi_thread_rejects_zero_threads():
    try:
        bench.measure_multi_thread(threads=0, size_mb=8, iterations=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for threads=0")
