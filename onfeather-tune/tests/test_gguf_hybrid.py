"""Coverage for architectures that broke the first version of the reader.

Every case here was found by running `of inspect` against a real Qwen3.5 27B
pulled from Ollama, not by imagining what a model might look like. It is a
hybrid SSM/attention model, it carries a vision encoder, it has a
speculative-decoding head, and it publishes `head_count_kv` as a per-layer
array. None of that fitted the assumptions the reader started with.
"""

from conftest import build_gguf

from onfeather_tune import gguf
from onfeather_tune.gguf import TensorRole

Q4_K, F32 = 12, 0


def hybrid_metadata(**overrides):
    base = {
        "general.architecture": "qwen35",
        "general.alignment": 32,
        "qwen35.block_count": 8,
        "qwen35.embedding_length": 5120,
        "qwen35.context_length": 262144,
        "qwen35.attention.head_count": 24,
        "qwen35.attention.key_length": 256,
        "qwen35.attention.value_length": 256,
    }
    base.update(overrides)
    return base


def hybrid_tensors():
    """Two attention layers among eight, plus vision and MTP submodels."""
    tensors = [("token_embd.weight", (5120, 1024), Q4_K)]
    for block in range(8):
        tensors.append((f"blk.{block}.attn_norm.weight", (5120,), F32))
        if block % 4 == 3:
            tensors.append((f"blk.{block}.attn_q.weight", (5120, 6144), Q4_K))
            tensors.append((f"blk.{block}.attn_output.weight", (6144, 5120), Q4_K))
        else:
            tensors.append((f"blk.{block}.ssm_conv1d.weight", (256, 6144), Q4_K))
            tensors.append((f"blk.{block}.ssm_out.weight", (6144, 5120), Q4_K))
            tensors.append((f"blk.{block}.ssm_a", (256,), F32))
    for block in range(4):
        tensors.append((f"v.blk.{block}.attn_q.weight", (1152, 1152), Q4_K))
        tensors.append((f"v.blk.{block}.mlp.linear_fc1.weight", (1152, 4304), Q4_K))
    tensors.append(("v.patch_embed.weight", (1152, 768), Q4_K))
    tensors.append(("mtp.fc.weight", (5120, 5120), Q4_K))
    tensors.append(("mtp.layers.0.attn_q.weight", (5120, 6144), Q4_K))
    return tensors


# -- classification -------------------------------------------------------


def test_state_space_tensors_are_recognised():
    assert gguf.classify("blk.3.ssm_conv1d.weight") is TensorRole.SSM
    assert gguf.classify("blk.3.ssm_out.weight") is TensorRole.SSM
    assert gguf.classify("blk.3.ssm_a") is TensorRole.SSM
    assert gguf.classify("blk.3.ssm_dt") is TensorRole.SSM


def test_state_space_layers_are_hot():
    """In a hybrid model they replace attention rather than supplementing it,
    so they run on every token and must not be offloaded."""
    assert TensorRole.SSM in gguf.HOT_ROLES


def test_vision_tensors_are_recognised_despite_attention_like_names():
    """`v.blk.0.attn_q` is spelled like the language model's attention but is
    idle for text-only prompts. Prefix wins over suffix."""
    assert gguf.classify("v.blk.0.attn_q.weight") is TensorRole.VISION
    assert gguf.classify("v.merger.linear_fc1.weight") is TensorRole.VISION
    assert gguf.classify("v.patch_embed.weight") is TensorRole.VISION
    assert gguf.classify("v.pos_embed.weight") is TensorRole.VISION


def test_mtp_tensors_are_recognised():
    assert gguf.classify("mtp.fc.weight") is TensorRole.MTP
    assert gguf.classify("mtp.layers.0.ffn_down.weight") is TensorRole.MTP
    assert gguf.classify("mtp.norm.weight") is TensorRole.MTP


def test_vision_and_mtp_are_cold():
    """Neither runs during plain text generation, so both are offloadable."""
    assert TensorRole.VISION not in gguf.HOT_ROLES
    assert TensorRole.MTP not in gguf.HOT_ROLES


def test_nothing_falls_through_to_other():
    """The 1.9 GiB that landed in `other` on the real model was the bug that
    started all of this."""
    model = gguf.read(build_gguf(hybrid_metadata(), hybrid_tensors()))
    unclassified = [t.name for t in model.tensors if t.role is TensorRole.OTHER]
    assert unclassified == []


# -- per-layer head_count_kv -----------------------------------------------------------


def test_array_head_count_kv_reads_the_attention_value():
    """Published as [0, 0, 0, 4, ...] with zeros on the state-space layers."""
    metadata = hybrid_metadata()
    metadata["qwen35.attention.head_count_kv"] = [0, 0, 0, 4] * 2
    model = gguf.read(build_gguf(metadata, hybrid_tensors()))

    assert model.head_count_kv == 4


def test_array_head_count_kv_counts_only_attention_layers():
    metadata = hybrid_metadata()
    metadata["qwen35.attention.head_count_kv"] = [0, 0, 0, 4] * 2
    model = gguf.read(build_gguf(metadata, hybrid_tensors()))

    assert model.block_count == 8
    assert model.attention_layer_count == 2


def test_scalar_head_count_kv_still_works():
    metadata = hybrid_metadata()
    metadata["qwen35.attention.head_count_kv"] = 4
    model = gguf.read(build_gguf(metadata, hybrid_tensors()))

    assert model.head_count_kv == 4
    assert model.attention_layer_count == model.block_count


def test_absent_head_count_kv_falls_back_to_head_count():
    model = gguf.read(build_gguf(hybrid_metadata(), hybrid_tensors()))
    assert model.head_count_kv == 24


# -- head dimension -------------------------------------------------------


def test_declared_key_length_beats_the_derived_head_dim():
    """5120 / 24 is 213, but the model declares 256. Deriving it understates
    the KV cache by a fifth."""
    model = gguf.read(build_gguf(hybrid_metadata(), hybrid_tensors()))
    assert model.head_dim == 256


def test_head_dim_falls_back_to_division_when_undeclared():
    metadata = hybrid_metadata()
    del metadata["qwen35.attention.key_length"]
    model = gguf.read(build_gguf(metadata, hybrid_tensors()))
    assert model.head_dim == 5120 // 24


# -- accounting -----------------------------------------------------------


def test_hot_and_cold_partition_a_multimodal_model():
    model = gguf.read(build_gguf(hybrid_metadata(), hybrid_tensors()))
    assert model.hot_bytes + model.cold_bytes == model.total_bytes


def test_vision_weights_count_as_offloadable():
    model = gguf.read(build_gguf(hybrid_metadata(), hybrid_tensors()))
    vision = sum(t.n_bytes for t in model.tensors if t.role is TensorRole.VISION)

    assert vision > 0
    assert model.cold_bytes >= vision


def test_text_generation_does_not_read_the_vision_encoder():
    model = gguf.read(build_gguf(hybrid_metadata(), hybrid_tensors()))
    assert model.active_bytes_per_token() < model.total_bytes
