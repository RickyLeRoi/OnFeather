import pytest

from onfeather_tune import estimate

GIB = 1024**3


def test_prediction_is_effective_bandwidth_over_active_bytes():
    result = estimate.decode_estimate(active_bytes=2_000_000_000, bandwidth_gbs=20.0)
    assert result.effective_bandwidth_gbs == pytest.approx(20.0 * estimate.WEIGHT_READ_FACTOR)
    assert result.predicted_tok_s == pytest.approx(20.0 * estimate.WEIGHT_READ_FACTOR / 2.0)


def test_read_factor_exceeds_one():
    """Weight streaming is pure sequential read; the STREAM `add` benchmark pays
    for writes and read-for-ownership, so it under-reports what inference gets."""
    assert estimate.WEIGHT_READ_FACTOR > 1.0
    result = estimate.decode_estimate(1_000_000_000, 20.0)
    assert result.effective_bandwidth_gbs > result.bandwidth_gbs


def test_uncertainty_band_brackets_the_prediction():
    result = estimate.decode_estimate(1_000_000_000, 20.0)
    assert result.low_tok_s < result.predicted_tok_s < result.high_tok_s


def test_halving_active_bytes_doubles_the_prediction():
    """The planner's core lever: shrink what is read per token."""
    slow = estimate.decode_estimate(2_000_000_000, 20.0)
    fast = estimate.decode_estimate(1_000_000_000, 20.0)
    assert fast.predicted_tok_s == pytest.approx(slow.predicted_tok_s * 2)


def test_zero_active_bytes_does_not_divide_by_zero():
    result = estimate.decode_estimate(0, 20.0)
    assert result.predicted_tok_s == float("inf")
    assert "not bandwidth-bound" in result.describe()


# -- calibration regression -----------------------------------------------

MEASURED = [
    ("qwen2.5-coder:7b-instruct", 4_681_000_000, 5.35),
    ("qwen2.5-coder:14b", 8_980_000_000, 2.79),
    # 20260725 RG A different architecture: hybrid state-space/attention, with vision.
    ("qwen3.6:27b", 16_220_000_000, 1.514),
]


@pytest.mark.parametrize(("name", "active_bytes", "measured"), MEASURED)
def test_prediction_reproduces_real_measurements(name, active_bytes, measured):
    result = estimate.decode_estimate(active_bytes, bandwidth_gbs=20.0)
    error = abs(result.predicted_tok_s - measured) / measured
    assert error < 0.05, f"{name}: predicted {result.predicted_tok_s:.2f}, measured {measured}"


@pytest.mark.parametrize(("name", "active_bytes", "measured"), MEASURED)
def test_measurements_fall_inside_the_uncertainty_band(name, active_bytes, measured):
    result = estimate.decode_estimate(active_bytes, bandwidth_gbs=20.0)
    assert result.low_tok_s <= measured <= result.high_tok_s, name


def test_all_calibration_runs_imply_the_same_bandwidth():
    """The finding that made this a calibration rather than a guess: models
    spanning a 3.5x range of size, across two unrelated architectures, agree on
    the implied bandwidth to within a couple of percent. That only happens if
    decode really is bandwidth-bound."""
    implied = [measured * active / 1e9 for _, active, measured in MEASURED]
    assert (max(implied) - min(implied)) / max(implied) < 0.03


def test_calibration_covers_a_wide_size_range():
    """Two nearby sizes could agree by coincidence; a 3.5x span cannot."""
    sizes = [active for _, active, _ in MEASURED]
    assert max(sizes) / min(sizes) > 3.0


# -- KV cache -------------------------------------------------------------


def qwen3_kv(context_length: int, **overrides) -> int:
    kwargs = {
        "layer_count": 48,
        "head_count_kv": 4,
        "head_dim": 64,
        "context_length": context_length,
    }
    kwargs.update(overrides)
    return estimate.kv_cache_bytes(**kwargs)


def test_kv_cache_grows_linearly_with_context():
    """A plan that fits at 4k can OOM at 32k -- the planner must see this."""
    assert qwen3_kv(32768) == pytest.approx(qwen3_kv(4096) * 8)


def test_gqa_shrinks_the_kv_cache():
    full = qwen3_kv(32768, head_count_kv=32)
    grouped = qwen3_kv(32768, head_count_kv=4)
    assert grouped == pytest.approx(full / 8)


def test_hybrid_model_charges_only_its_attention_layers():
    """Qwen3.5 runs 16 attention layers among 64. Charging all 64 for KV would
    overstate the cache fourfold and push the planner to offload weights that
    would have fitted."""
    all_layers = qwen3_kv(32768, layer_count=64)
    hybrid = qwen3_kv(32768, layer_count=16)
    assert hybrid == pytest.approx(all_layers / 4)


def test_kv_cache_size_is_plausible():
    """48 layers x 4 KV heads x 64 dim x 2 (K and V) x 2 bytes x 32768 tokens."""
    assert qwen3_kv(32768) == 48 * 4 * 64 * 2 * 2 * 32768


def test_quantised_kv_cache_is_smaller():
    f16 = qwen3_kv(32768)
    q8 = qwen3_kv(32768, bytes_per_element=1)
    assert q8 == f16 // 2


def test_kv_cache_rejects_zero_dimensions():
    with pytest.raises(ValueError, match="must be positive"):
        qwen3_kv(0)
