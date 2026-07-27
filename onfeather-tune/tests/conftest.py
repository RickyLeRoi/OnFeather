"""Synthetic GGUF construction, so the suite needs no multi-gigabyte fixture."""

from __future__ import annotations

import struct
from io import BytesIO

from onfeather_tune.gguf import DEFAULT_ALIGNMENT, ValueType


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _value(value: object) -> bytes:
    """Encode a metadata value, inferring its GGUF type from the Python type."""
    if isinstance(value, str):
        return struct.pack("<I", ValueType.STRING) + _string(value)
    if isinstance(value, bool):
        return struct.pack("<I", ValueType.BOOL) + struct.pack("<?", value)
    if isinstance(value, int):
        return struct.pack("<I", ValueType.UINT32) + struct.pack("<I", value)
    if isinstance(value, float):
        return struct.pack("<I", ValueType.FLOAT32) + struct.pack("<f", value)
    if isinstance(value, list):
        body = b"".join(_value(item)[4:] for item in value)
        element_type = ValueType.STRING if value and isinstance(value[0], str) else ValueType.UINT32
        return (
            struct.pack("<I", ValueType.ARRAY)
            + struct.pack("<I", element_type)
            + struct.pack("<Q", len(value))
            + body
        )
    raise TypeError(f"unsupported metadata value: {value!r}")


def build_gguf(
    metadata: dict[str, object],
    tensors: list[tuple[str, tuple[int, ...], int]],
    *,
    version: int = 3,
    alignment: int = DEFAULT_ALIGNMENT,
) -> BytesIO:
    """Build a header-complete GGUF file in memory.

    `tensors` entries are (name, dimensions, ggml_type_id). Offsets are assigned
    sequentially; no tensor data is written, since the reader never reads any.
    """
    body = b""
    for name, dimensions, type_id in tensors:
        body += _string(name)
        body += struct.pack("<I", len(dimensions))
        body += b"".join(struct.pack("<Q", dim) for dim in dimensions)
        body += struct.pack("<I", type_id)
        body += struct.pack("<Q", 0)

    header = b"GGUF"
    header += struct.pack("<I", version)
    header += struct.pack("<Q", len(tensors))
    header += struct.pack("<Q", len(metadata))
    for key, value in metadata.items():
        header += _string(key) + _value(value)

    return BytesIO(header + body)


def moe_metadata(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "general.architecture": "qwen3moe",
        "general.alignment": DEFAULT_ALIGNMENT,
        "qwen3moe.block_count": 2,
        "qwen3moe.embedding_length": 2048,
        "qwen3moe.context_length": 32768,
        "qwen3moe.attention.head_count": 32,
        "qwen3moe.attention.head_count_kv": 4,
        "qwen3moe.expert_count": 128,
        "qwen3moe.expert_used_count": 8,
    }
    base.update(overrides)
    return base


def moe_tensors(blocks: int = 2, q4k: int = 12, f32: int = 0) -> list[tuple[str, tuple[int, ...], int]]:
    """A MoE layer stack shaped like Qwen3-30B-A3B, scaled down."""
    tensors: list[tuple[str, tuple[int, ...], int]] = [
        ("token_embd.weight", (2048, 151936), q4k),
        ("output_norm.weight", (2048,), f32),
        ("output.weight", (2048, 151936), q4k),
    ]
    for block in range(blocks):
        tensors += [
            (f"blk.{block}.attn_norm.weight", (2048,), f32),
            (f"blk.{block}.attn_q.weight", (2048, 4096), q4k),
            (f"blk.{block}.attn_k.weight", (2048, 512), q4k),
            (f"blk.{block}.attn_v.weight", (2048, 512), q4k),
            (f"blk.{block}.attn_output.weight", (4096, 2048), q4k),
            (f"blk.{block}.ffn_norm.weight", (2048,), f32),
            (f"blk.{block}.ffn_gate_inp.weight", (2048, 128), f32),
            (f"blk.{block}.ffn_gate_exps.weight", (2048, 768, 128), q4k),
            (f"blk.{block}.ffn_up_exps.weight", (2048, 768, 128), q4k),
            (f"blk.{block}.ffn_down_exps.weight", (768, 2048, 128), q4k),
        ]
    return tensors
