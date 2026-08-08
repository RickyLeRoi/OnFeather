import re

import pytest
from conftest import build_gguf, moe_metadata, moe_tensors

from onfeather_tune import gguf, plan as plan_module
from onfeather_tune.model import CpuInfo, GpuInfo, HardwareProfile, MemoryInfo, PlatformInfo
from onfeather_tune.plan import PlanError, make_plan

GIB = 1024**3


def make_profile(vram_gib: float = 4.0, *, unified: bool = False, bandwidth: float = 20.0):
    gpus = []
    if vram_gib:
        gpus.append(
            GpuInfo(
                name="Radeon Pro 5500M",
                vendor="amd",
                total_bytes=int(vram_gib * GIB),
                free_bytes=int(vram_gib * GIB),
                unified_memory=unified,
            )
        )
    return HardwareProfile(
        schema_version=1,
        platform=PlatformInfo(system="Darwin", release="25.5.0", machine="x86_64", python="3.11"),
        cpu=CpuInfo(model="i9-9880H", physical_cores=8, logical_cores=16, isa=["avx2"]),
        memory=MemoryInfo(
            total_bytes=32 * GIB,
            available_bytes=17 * GIB,
            bandwidth_single_gbs=12.3,
            bandwidth_multi_gbs=bandwidth,
        ),
        gpus=gpus,
    )


@pytest.fixture
def model():
    return gguf.read(build_gguf(moe_metadata(), moe_tensors(blocks=8)))


# -- budget ---------------------------------------------------------------


def test_hot_tensors_are_paid_before_experts(model):
    """A byte spent on attention serves every token; a byte spent on an expert
    serves one token in sixteen. Hot always wins the budget."""
    plan = make_plan(model, make_profile(vram_gib=4.0))

    assert not plan.hot_on_cpu
    assert plan.gpu_bytes_used <= plan.gpu_bytes_available


def test_plan_never_exceeds_the_budget(model):
    for vram in (0.5, 1.0, 2.0, 4.0, 8.0, 24.0):
        plan = make_plan(model, make_profile(vram_gib=vram))
        assert plan.gpu_bytes_used <= plan.gpu_bytes_available, f"overflowed at {vram} GiB"


def test_more_vram_keeps_more_experts_resident(model):
    small = make_plan(model, make_profile(vram_gib=2.0))
    large = make_plan(model, make_profile(vram_gib=8.0))
    assert len(large.cpu_expert_layers) < len(small.cpu_expert_layers)


def test_large_gpu_keeps_everything(model):
    plan = make_plan(model, make_profile(vram_gib=64.0))
    assert plan.cpu_expert_layers == []
    assert plan.override_tensor_patterns() == []
    assert plan.active_bytes_from_ram == 0


def test_tiny_gpu_falls_back_to_cpu(model):
    plan = make_plan(model, make_profile(vram_gib=0.5))
    assert plan.hot_on_cpu
    assert "-ngl" in plan.llama_args()
    assert plan.llama_args()[plan.llama_args().index("-ngl") + 1] == "0"


def test_no_gpu_falls_back_to_cpu(model):
    plan = make_plan(model, make_profile(vram_gib=0))
    assert plan.hot_on_cpu


def test_unified_memory_gpu_is_not_counted(model):
    """Moving a tensor to unified 'VRAM' buys no bandwidth: it is the same DRAM
    the CPU already reads, so the hot/cold split would be meaningless."""
    plan = make_plan(model, make_profile(vram_gib=16.0, unified=True))
    assert plan.hot_on_cpu


def occupied_profile(total_gib: float, free_gib: float | None):
    """A card whose free VRAM is measured separately from its size."""
    profile = make_profile(vram_gib=total_gib)
    gpu = profile.gpus[0]
    profile.gpus[0] = GpuInfo(
        name=gpu.name,
        vendor=gpu.vendor,
        total_bytes=gpu.total_bytes,
        free_bytes=None if free_gib is None else int(free_gib * GIB),
        unified_memory=False,
    )
    return profile


def test_a_full_gpu_is_measured_not_assumed(model):
    """Zero free is a measurement. Planning against the whole card from it is the
    one failure this tool exists to prevent."""
    plan = make_plan(model, occupied_profile(total_gib=16.0, free_gib=0.0))
    assert plan.hot_on_cpu


def test_unmeasured_free_vram_still_plans_against_the_card(model):
    """None means nobody looked, which is different from looking and finding nothing."""
    plan = make_plan(model, occupied_profile(total_gib=16.0, free_gib=None))
    assert not plan.hot_on_cpu


def test_the_biggest_gpu_is_the_one_with_the_most_free_vram(model):
    """A busy 16 GiB card must not outrank an idle 8 GiB one."""
    profile = occupied_profile(total_gib=16.0, free_gib=0.0)
    profile.gpus.append(
        GpuInfo(name="idle", vendor="nvidia", total_bytes=8 * GIB, free_bytes=8 * GIB)
    )
    plan = make_plan(model, profile)
    assert not plan.hot_on_cpu


def test_planning_on_a_shared_gpu_says_so(model, capsys):
    make_plan(model, occupied_profile(total_gib=16.0, free_gib=4.0))
    assert "something else" in capsys.readouterr().err


def test_planning_on_an_idle_gpu_stays_quiet(model, capsys):
    make_plan(model, make_profile(vram_gib=16.0))
    assert capsys.readouterr().err == ""


def test_reserve_is_held_back(model):
    generous = make_plan(model, make_profile(vram_gib=4.0), reserve_bytes=0)
    cautious = make_plan(model, make_profile(vram_gib=4.0), reserve_bytes=1 * GIB)
    assert cautious.gpu_bytes_available < generous.gpu_bytes_available


# -- KV cache interaction -------------------------------------------------


def test_longer_context_squeezes_out_experts(model):
    """The failure mode the planner exists to prevent: a split that fits at 4k
    and OOMs at 32k."""
    short = make_plan(model, make_profile(vram_gib=4.0), context_length=4096)
    long = make_plan(model, make_profile(vram_gib=4.0), context_length=32768)

    assert long.kv_cache_bytes > short.kv_cache_bytes
    assert len(long.cpu_expert_layers) >= len(short.cpu_expert_layers)
    assert long.gpu_bytes_used <= long.gpu_bytes_available


def test_context_defaults_to_model_maximum(model):
    assert make_plan(model, make_profile()).context_length == model.context_length


# -- generated arguments --------------------------------------------------


def test_override_pattern_targets_only_routed_experts(model):
    pattern = make_plan(model, make_profile(vram_gib=2.0)).override_tensor_patterns()[0]
    expression, buffer = pattern.rsplit("=", 1)
    assert buffer == "CPU"

    compiled = re.compile(expression)
    assert compiled.search("blk.7.ffn_gate_exps.weight")
    # 20260725 RG Shared experts and the router run on every token.
    assert not compiled.search("blk.7.ffn_gate_shexp.weight")
    assert not compiled.search("blk.7.ffn_gate_inp.weight")
    assert not compiled.search("blk.7.attn_q.weight")


def test_pattern_matches_exactly_the_offloaded_layers(model):
    plan = make_plan(model, make_profile(vram_gib=2.0))
    compiled = re.compile(plan.override_tensor_patterns()[0].rsplit("=", 1)[0])

    for layer in range(model.block_count):
        name = f"blk.{layer}.ffn_up_exps.weight"
        expected = layer in plan.cpu_expert_layers
        assert bool(compiled.search(name)) is expected, f"layer {layer} misplaced"


def test_whole_model_offload_uses_a_compact_pattern(model):
    plan = make_plan(model, make_profile(vram_gib=1.2))
    if len(plan.cpu_expert_layers) == model.block_count and not plan.hot_on_cpu:
        assert r"blk\.\d+\." in plan.override_tensor_patterns()[0]


def test_layer_pattern_rejects_empty_set():
    with pytest.raises(PlanError):
        plan_module._layers_pattern([], 8)


def test_command_is_runnable_shape(model):
    command = make_plan(model, make_profile(vram_gib=2.0)).command(model_path="m.gguf")
    assert command.startswith("llama-server -m m.gguf")
    assert "-ngl" in command
    assert "-ot" in command


# -- speed estimate -------------------------------------------------------


def test_only_ram_resident_experts_count_toward_the_estimate(model):
    """VRAM-resident weights are read at GPU bandwidth and are not the
    bottleneck, so they must not inflate the per-token RAM traffic."""
    plan = make_plan(model, make_profile(vram_gib=4.0))
    assert plan.active_bytes_from_ram < model.active_bytes_per_token()


def test_offloading_less_predicts_more_speed(model):
    small = make_plan(model, make_profile(vram_gib=2.0))
    large = make_plan(model, make_profile(vram_gib=6.0))
    assert large.estimate.predicted_tok_s > small.estimate.predicted_tok_s


def test_estimate_absent_without_a_bandwidth_measurement(model):
    profile = make_profile()
    profile.memory.bandwidth_multi_gbs = None
    assert make_plan(model, profile).estimate is None


def test_summary_mentions_the_command(model):
    text = plan_module.summarise(make_plan(model, make_profile(vram_gib=2.0)))
    assert "COMMAND" in text
    assert "llama-server" in text


# -- containers -----------------------------------------------------------


def test_a_restricted_machine_gets_an_explicit_thread_count(model):
    """llama.cpp defaults to the machine's cores, which a container does not own.

    On a 128-core host entitled to 18, decode measured 26.4 tok/s at 18 threads,
    12.8 at 36 and 6.7 at 64. Saying nothing costs a 4x slowdown silently.
    """
    profile = make_profile()
    profile.cpu.physical_cores = 64
    profile.cpu.logical_cores = 128
    profile.cpu.usable_cores = 18

    built = make_plan(model, profile)
    assert built.threads == 18
    assert "-t 18" in built.command()


def test_an_ordinary_machine_gets_no_thread_flag(model):
    """Where the default is right, adding `-t` is noise in a command people copy."""
    profile = make_profile()
    profile.cpu.usable_cores = 16

    built = make_plan(model, profile)
    assert built.threads is None
    assert " -t " not in built.command()


def test_an_undetectable_entitlement_stays_silent(model):
    """Windows has no affinity mask and no cgroups; guessing would be worse."""
    profile = make_profile()
    profile.cpu.usable_cores = None

    assert make_plan(model, profile).threads is None


def test_a_gpu_heavy_plan_admits_its_estimate_is_optimistic(model):
    """The bandwidth model only describes a bandwidth-bound machine.

    Measured on a P100 against Qwen3-30B-A3B: predicted 71.1 tok/s, measured
    32.87, with a tenth of the active weights coming from RAM. Printing that
    figure bare is worse than printing nothing.
    """
    profile = make_profile(vram_gib=16.0)
    built = make_plan(model, profile)

    assert built.ram_bound_fraction < plan_module.RAM_BOUND_FLOOR
    assert "optimistic" in plan_module.summarise(built)


def test_a_ram_bound_plan_says_nothing_extra(model):
    """Where the model is validated, a caveat would just be noise."""
    profile = make_profile(vram_gib=0)
    built = make_plan(model, profile)

    assert built.ram_bound_fraction == 1.0
    assert "optimistic" not in plan_module.summarise(built)
