"""First-order performance estimates.

Decode on CPU-resident weights is bandwidth-bound: the arithmetic per weight is
trivial, so the memory bus, not the ALUs, sets the pace. That gives a ceiling

    tok/s  =  bandwidth / active bytes per token

which no configuration can beat and a good one approaches. The planner uses it
to rank candidate plans before spending minutes benchmarking them.
"""

from __future__ import annotations

from dataclasses import dataclass

#: 20260726 ** RG Weight streaming is pure read, so it beats the STREAM add benchmark. Calibrated: see calibration/.
WEIGHT_READ_FACTOR = 1.25

#: 20260726 ** RG How much of the effective read bandwidth llama.cpp actually reaches.
DEFAULT_EFFICIENCY = 1.0

#: 20260726 ** RG Multipliers bracketing the central prediction.
UNCERTAINTY_LOW = 0.85
UNCERTAINTY_HIGH = 1.10


@dataclass(frozen=True)
class DecodeEstimate:
    active_bytes: int
    bandwidth_gbs: float
    """STREAM `add` bandwidth as reported by `of probe`, not the effective
    read bandwidth: the conversion happens here so callers cannot forget it."""
    efficiency: float = DEFAULT_EFFICIENCY
    read_factor: float = WEIGHT_READ_FACTOR

    @property
    def effective_bandwidth_gbs(self) -> float:
        """Bandwidth available to pure sequential weight reads."""
        return self.bandwidth_gbs * self.read_factor

    @property
    def predicted_tok_s(self) -> float:
        """Central estimate: effective bandwidth over bytes read per token."""
        if self.active_bytes <= 0:
            return float("inf")
        return self.effective_bandwidth_gbs * self.efficiency * 1e9 / self.active_bytes

    @property
    def low_tok_s(self) -> float:
        return self.predicted_tok_s * UNCERTAINTY_LOW

    @property
    def high_tok_s(self) -> float:
        return self.predicted_tok_s * UNCERTAINTY_HIGH

    def describe(self) -> str:
        if self.active_bytes <= 0:
            return "not bandwidth-bound"
        return f"{self.predicted_tok_s:.1f} tok/s ({self.low_tok_s:.1f}-{self.high_tok_s:.1f})"


def decode_estimate(
    active_bytes: int,
    bandwidth_gbs: float,
    efficiency: float = DEFAULT_EFFICIENCY,
    read_factor: float = WEIGHT_READ_FACTOR,
) -> DecodeEstimate:
    return DecodeEstimate(active_bytes, bandwidth_gbs, efficiency, read_factor)


def kv_cache_bytes(
    *,
    layer_count: int,
    head_count_kv: int,
    head_dim: int,
    context_length: int,
    bytes_per_element: int = 2,
) -> int:
    """Size of the KV cache at a given context length.

    The cache competes with weights for the same VRAM and grows linearly with
    context, so a plan that fits at 4k can OOM at 32k. GQA shrinks it by the
    ratio of KV heads to query heads, which is why head_count_kv matters here.

    `layer_count` is the number of layers that actually hold a cache, not the
    model's depth: hybrid models interleave state-space layers that carry a
    fixed-size state instead, and charging those for KV inflates the estimate
    by the ratio between the two kinds.
    """
    if not all((layer_count, head_count_kv, head_dim, context_length)):
        raise ValueError("all model dimensions must be positive")

    per_token_per_layer = 2 * head_count_kv * head_dim * bytes_per_element
    return per_token_per_layer * layer_count * context_length
