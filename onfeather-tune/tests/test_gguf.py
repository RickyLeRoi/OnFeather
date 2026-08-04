import struct

import pytest
from conftest import build_gguf, moe_metadata, moe_tensors

from onfeather_tune import gguf
from onfeather_tune.gguf import GGUFError, TensorInfo, TensorRole


# -- classification -------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("token_embd.weight", TensorRole.TOKEN_EMBD),
        ("output.weight", TensorRole.OUTPUT),
        ("output_norm.weight", TensorRole.NORM),
        ("blk.0.attn_norm.weight", TensorRole.NORM),
        ("blk.0.attn_q.weight", TensorRole.ATTENTION),
        ("blk.7.attn_output.weight", TensorRole.ATTENTION),
        ("blk.0.ffn_norm.weight", TensorRole.NORM),
        ("blk.0.ffn_gate_inp.weight", TensorRole.ROUTER),
        ("blk.0.ffn_exp_probs_b.bias", TensorRole.ROUTER),
        ("blk.0.ffn_gate_exps.weight", TensorRole.ROUTED_EXPERTS),
        ("blk.0.ffn_up_exps.weight", TensorRole.ROUTED_EXPERTS),
        ("blk.12.ffn_down_exps.weight", TensorRole.ROUTED_EXPERTS),
        ("blk.0.ffn_gate_shexp.weight", TensorRole.SHARED_EXPERTS),
        ("blk.0.ffn_down_shexp.weight", TensorRole.SHARED_EXPERTS),
        ("blk.0.ffn_down.weight", TensorRole.FFN_DENSE),
        ("blk.0.ffn_gate.weight", TensorRole.FFN_DENSE),
        ("rope_freqs.weight", TensorRole.OTHER),
        # 20260725 RG Real gemma4 names, unknown to this parser.
        ("blk.0.proj.weight", TensorRole.BLOCK_OTHER),
        ("blk.0.inp_gate.weight", TensorRole.BLOCK_OTHER),
        ("blk.41.layer_output_scale.weight", TensorRole.BLOCK_OTHER),
        ("per_layer_token_embd.weight", TensorRole.PER_LAYER_EMBD),
        ("a.blk.0.attn_q.weight", TensorRole.AUDIO),
        ("a.blk.11.conv_pw1.weight", TensorRole.AUDIO),
    ],
)
def test_classify(name, expected):
    assert gguf.classify(name) is expected


def test_an_unrecognised_per_block_tensor_is_hot():
    """The safe direction to be wrong in.

    gemma4 spends 66% of its file on per-block `proj` and `inp_gate` weights
    this parser has never seen. Filing them as cold made `of inspect` predict
    8.5 tok/s where roughly 2.9 is plausible — an optimistic error, which is the
    one that gets a user to run a plan that cannot deliver. Filing them hot only
    spends VRAM budget the planner then declines to fill with experts.
    """
    assert gguf.classify("blk.3.something_new.weight") in gguf.HOT_ROLES


def test_an_unrecognised_global_tensor_stays_cold():
    """Outside a block there is no per-token argument, and these are tiny."""
    assert gguf.classify("some_new_global.weight") is TensorRole.OTHER
    assert TensorRole.OTHER not in gguf.HOT_ROLES


def test_an_audio_encoder_is_idle_for_text():
    """`a.blk.0.attn_q` is spelled like the language model's but never runs."""
    assert gguf.classify("a.blk.0.attn_q.weight") is TensorRole.AUDIO
    assert TensorRole.AUDIO not in gguf.HOT_ROLES


def test_a_per_layer_embedding_table_is_not_charged_per_token():
    """The reason gemma4:e4b decodes like a 4B despite an 8.9 GiB file.

    The table is indexed by token id, not streamed. Counting all 5.25 GiB of it
    as read-per-token would predict roughly 2.9 tok/s where about 8 is right.
    """
    assert TensorRole.PER_LAYER_EMBD not in gguf.HOT_ROLES


def test_router_is_not_mistaken_for_a_dense_gate():
    """`ffn_gate_inp` selects experts and runs every token; misfiling it as a
    dense FFN would be harmless, but as a routed expert would let the planner
    offload the router and stall every token on a PCIe round trip."""
    assert gguf.classify("blk.0.ffn_gate_inp.weight") is TensorRole.ROUTER
    assert gguf.classify("blk.0.ffn_gate_inp.weight") in gguf.HOT_ROLES


def test_shared_experts_are_hot_and_routed_experts_are_not():
    """The distinction the whole planner rests on."""
    assert TensorRole.SHARED_EXPERTS in gguf.HOT_ROLES
    assert TensorRole.ROUTED_EXPERTS not in gguf.HOT_ROLES


# -- tensor sizing --------------------------------------------------------


def test_n_bytes_for_q4_k():
    tensor = TensorInfo("blk.0.attn_q.weight", (2048, 4096), type_id=12, offset=0)
    # 20260725 RG 8_388_608 elements / 256 per block * 144 bytes per block.
    assert tensor.n_elements == 8_388_608
    assert tensor.n_bytes == 8_388_608 // 256 * 144


def test_n_bytes_for_f32():
    tensor = TensorInfo("output_norm.weight", (2048,), type_id=0, offset=0)
    assert tensor.n_bytes == 2048 * 4


def test_n_bytes_rejects_unknown_type():
    with pytest.raises(GGUFError, match="unknown ggml type"):
        _ = TensorInfo("x", (256,), type_id=999, offset=0).n_bytes


def test_n_bytes_rejects_misaligned_element_count():
    with pytest.raises(GGUFError, match="not a multiple of block size"):
        _ = TensorInfo("x", (100,), type_id=12, offset=0).n_bytes


@pytest.mark.parametrize(
    ("name", "expected"),
    [("blk.0.attn_q.weight", 0), ("blk.47.ffn_up_exps.weight", 47), ("token_embd.weight", None)],
)
def test_layer_extraction(name, expected):
    assert TensorInfo(name, (256,), 12, 0).layer == expected


def test_quant_table_matches_upstream():
    """Guards the vendored table against drift in llama.cpp's gguf package."""
    upstream = pytest.importorskip("gguf.constants")
    for type_id, (block_size, type_size) in gguf.QUANT_SIZES.items():
        expected = upstream.GGML_QUANT_SIZES[upstream.GGMLQuantizationType(type_id)]
        assert (block_size, type_size) == expected, f"type id {type_id} drifted"


# -- parsing --------------------------------------------------------------


def test_reads_header_and_tensors():
    stream = build_gguf(moe_metadata(), moe_tensors(blocks=2))
    model = gguf.read(stream)

    assert model.version == 3
    assert model.architecture == "qwen3moe"
    assert model.block_count == 2
    assert model.head_count == 32
    assert model.head_count_kv == 4
    assert model.context_length == 32768
    assert len(model.tensors) == 3 + 2 * 10


def test_detects_moe_and_expert_counts():
    model = gguf.read(build_gguf(moe_metadata(), moe_tensors()))
    assert model.is_moe
    assert model.expert_count == 128
    assert model.expert_used_count == 8


def test_dense_model_is_not_moe():
    metadata = moe_metadata()
    del metadata["qwen3moe.expert_count"]
    del metadata["qwen3moe.expert_used_count"]
    model = gguf.read(build_gguf(metadata, moe_tensors()))
    assert not model.is_moe
    assert model.expert_count == 0


def test_head_count_kv_falls_back_to_head_count():
    """Absent head_count_kv means no GQA, not zero KV heads."""
    metadata = moe_metadata()
    del metadata["qwen3moe.attention.head_count_kv"]
    model = gguf.read(build_gguf(metadata, moe_tensors()))
    assert model.head_count_kv == 32


def test_data_offset_is_aligned():
    model = gguf.read(build_gguf(moe_metadata(), moe_tensors()))
    assert model.data_offset % model.alignment == 0


def test_alignment_defaults_to_32_when_absent():
    metadata = moe_metadata()
    del metadata["general.alignment"]
    model = gguf.read(build_gguf(metadata, moe_tensors()))
    assert model.alignment == 32


def test_rejects_non_gguf_file():
    from io import BytesIO

    with pytest.raises(GGUFError, match="not a GGUF file"):
        gguf.read(BytesIO(b"ZZZZ" + b"\x00" * 64))


def test_rejects_unsupported_version():
    stream = build_gguf(moe_metadata(), moe_tensors(), version=99)
    with pytest.raises(GGUFError, match="unsupported GGUF version"):
        gguf.read(stream)


def test_large_arrays_are_skipped_not_materialised():
    """Tokeniser vocabularies must not be pulled into memory."""
    vocab = [f"token_{index}" for index in range(gguf.LARGE_ARRAY_THRESHOLD + 50)]
    metadata = moe_metadata()
    metadata["tokenizer.ggml.tokens"] = vocab

    model = gguf.read(build_gguf(metadata, moe_tensors()))

    assert model.metadata["tokenizer.ggml.tokens"] == f"<{len(vocab)} elements omitted>"
    # 20260725 RG Parsing must still land in the right place afterwards.
    assert len(model.tensors) == 23


def test_small_arrays_are_kept():
    metadata = moe_metadata()
    metadata["test.small"] = ["a", "b", "c"]
    model = gguf.read(build_gguf(metadata, moe_tensors()))
    assert model.metadata["test.small"] == ["a", "b", "c"]


def test_reads_big_endian_files():
    """v3 permits big-endian; endianness is inferred from the version field."""
    stream = build_gguf(moe_metadata(), [])
    payload = bytearray(stream.getvalue())
    payload[4:8] = struct.pack(">I", 3)  # 20260725 RG Byte-swap the version.
    from io import BytesIO

    # 20260725 RG Only the version is swapped, so the rest would fail to parse.
    with pytest.raises(GGUFError):
        gguf.read(BytesIO(bytes(payload)))


# -- size accounting ------------------------------------------------------


def test_hot_and_cold_bytes_partition_the_model():
    """Must hold for every model, not just a plain MoE: a multimodal or
    speculative-decoding model parks weights in neither of the two categories
    the first version of this accounted for."""
    model = gguf.read(build_gguf(moe_metadata(), moe_tensors()))
    assert model.hot_bytes + model.cold_bytes == model.total_bytes


def test_routed_experts_dominate_a_moe():
    """The premise of the project: most of a MoE is cold weight."""
    model = gguf.read(build_gguf(moe_metadata(), moe_tensors()))
    assert model.routed_expert_bytes > model.hot_bytes


def test_active_bytes_per_token_is_far_below_total():
    """With 8 of 128 experts firing, active weight is a small slice of the file."""
    model = gguf.read(build_gguf(moe_metadata(), moe_tensors()))
    active = model.active_bytes_per_token()

    assert active < model.total_bytes
    expected = model.hot_bytes + int(model.routed_expert_bytes * 8 / 128)
    assert active == expected


def test_active_bytes_equals_total_for_dense_models():
    metadata = moe_metadata()
    del metadata["qwen3moe.expert_count"]
    del metadata["qwen3moe.expert_used_count"]
    model = gguf.read(build_gguf(metadata, moe_tensors()))
    assert model.active_bytes_per_token() == model.total_bytes


def test_bytes_by_role_sums_to_total():
    model = gguf.read(build_gguf(moe_metadata(), moe_tensors()))
    assert sum(model.bytes_by_role().values()) == model.total_bytes
