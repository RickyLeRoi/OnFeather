"""Cross-validation against llama.cpp's reference implementation.

test_gguf.py builds its fixtures with our own writer, so a misreading of the
spec would be invisible: reader and writer would share the mistake. Here the
file is produced by the official `gguf` package and parsed by us, which catches
exactly that class of error.
"""

from __future__ import annotations

import numpy as np
import pytest

from onfeather_tune import gguf as of_gguf
from onfeather_tune.gguf import TensorRole

writer_module = pytest.importorskip("gguf")


@pytest.fixture(scope="module")
def reference_file(tmp_path_factory) -> str:
    """A small but structurally faithful MoE, written by the reference writer."""
    path = tmp_path_factory.mktemp("gguf") / "reference.gguf"

    writer = writer_module.GGUFWriter(str(path), "qwen3moe")
    writer.add_block_count(2)
    writer.add_context_length(32768)
    writer.add_embedding_length(256)
    writer.add_head_count(8)
    writer.add_head_count_kv(2)
    writer.add_expert_count(64)
    writer.add_expert_used_count(4)

    writer.add_tensor("token_embd.weight", np.zeros((512, 256), dtype=np.float32))
    writer.add_tensor("output_norm.weight", np.zeros((256,), dtype=np.float32))
    for block in range(2):
        writer.add_tensor(f"blk.{block}.attn_norm.weight", np.zeros((256,), dtype=np.float32))
        writer.add_tensor(f"blk.{block}.attn_q.weight", np.zeros((256, 256), dtype=np.float32))
        writer.add_tensor(f"blk.{block}.attn_k.weight", np.zeros((64, 256), dtype=np.float32))
        writer.add_tensor(f"blk.{block}.ffn_gate_inp.weight", np.zeros((64, 256), dtype=np.float32))
        writer.add_tensor(
            f"blk.{block}.ffn_gate_exps.weight", np.zeros((64, 128, 256), dtype=np.float32)
        )
        writer.add_tensor(
            f"blk.{block}.ffn_down_exps.weight", np.zeros((64, 256, 128), dtype=np.float32)
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return str(path)


def test_parses_a_reference_written_file(reference_file):
    model = of_gguf.read(reference_file)

    assert model.version == 3
    assert model.architecture == "qwen3moe"
    assert model.block_count == 2
    assert model.embedding_length == 256
    assert model.head_count == 8
    assert model.head_count_kv == 2
    assert model.context_length == 32768
    assert model.expert_count == 64
    assert model.expert_used_count == 4
    assert model.is_moe


def test_metadata_keys_match_our_expectations(reference_file):
    """Pins the exact key spelling the reference writer emits, so a rename
    upstream fails here rather than silently returning None from arch_key."""
    keys = of_gguf.read(reference_file).metadata.keys()

    for key in (
        "general.architecture",
        "qwen3moe.block_count",
        "qwen3moe.embedding_length",
        "qwen3moe.context_length",
        "qwen3moe.attention.head_count",
        "qwen3moe.attention.head_count_kv",
        "qwen3moe.expert_count",
        "qwen3moe.expert_used_count",
    ):
        assert key in keys


def test_tensor_names_and_shapes_survive_the_round_trip(reference_file):
    tensors = {t.name: t for t in of_gguf.read(reference_file).tensors}

    assert len(tensors) == 2 + 2 * 6
    # 20260725 RG GGUF stores dimensions in reverse of NumPy's order.
    assert tensors["token_embd.weight"].dimensions == (256, 512)
    assert tensors["blk.0.ffn_gate_exps.weight"].dimensions == (256, 128, 64)
    assert tensors["blk.0.ffn_gate_exps.weight"].role is TensorRole.ROUTED_EXPERTS
    assert tensors["blk.0.ffn_gate_inp.weight"].role is TensorRole.ROUTER


def test_computed_sizes_match_the_real_file(reference_file):
    """The end-to-end guarantee: our byte arithmetic reproduces actual file
    layout, which is what the planner will budget VRAM against."""
    import os

    model = of_gguf.read(reference_file)

    for tensor in model.tensors:
        expected = tensor.n_elements * 4
        assert tensor.n_bytes == expected, tensor.name

    # 20260725 RG Header, tensor table and data must account for the whole file.
    assert model.data_offset + model.total_bytes == os.path.getsize(reference_file)


def test_offsets_are_contiguous_and_aligned(reference_file):
    model = of_gguf.read(reference_file)

    cursor = 0
    for tensor in model.tensors:
        assert tensor.offset == cursor, f"{tensor.name} is not where we predicted"
        assert tensor.offset % model.alignment == 0
        cursor += tensor.n_bytes
        cursor += (model.alignment - cursor % model.alignment) % model.alignment


def test_active_bytes_below_total_on_a_real_moe(reference_file):
    model = of_gguf.read(reference_file)
    assert model.active_bytes_per_token() < model.total_bytes
