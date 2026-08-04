"""Header-only GGUF reader.

The planner needs tensor shapes, types and sizes -- never the weights. Reading
only the header keeps `of plan` instant on a 30 GB file, and leaves the door open
to planning against a remote model over HTTP range requests before downloading
it.

This deliberately does not depend on the official `gguf` package at runtime:
that package memory-maps the whole file to expose data we never touch. The
quantisation table below is vendored instead, and `tests/test_gguf.py`
cross-checks it against `gguf` when it is installed, so drift gets caught.

Format reference: GGUF v3 (ggml-org/ggml, docs/gguf.md).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from math import prod
from pathlib import Path
from typing import BinaryIO

GGUF_MAGIC = b"GGUF"
SUPPORTED_VERSIONS = (2, 3)
DEFAULT_ALIGNMENT = 32

# 20260725 RG Long arrays are walked, not materialised.
LARGE_ARRAY_THRESHOLD = 1024


class ValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


_SCALAR_FORMATS: dict[int, tuple[str, int]] = {
    ValueType.UINT8: ("B", 1),
    ValueType.INT8: ("b", 1),
    ValueType.UINT16: ("H", 2),
    ValueType.INT16: ("h", 2),
    ValueType.UINT32: ("I", 4),
    ValueType.INT32: ("i", 4),
    ValueType.FLOAT32: ("f", 4),
    ValueType.BOOL: ("?", 1),
    ValueType.UINT64: ("Q", 8),
    ValueType.INT64: ("q", 8),
    ValueType.FLOAT64: ("d", 8),
}

# 20260725 RG (block_size, type_size) per ggml type id.
QUANT_SIZES: dict[int, tuple[int, int]] = {
    0: (1, 4),  # F32
    1: (1, 2),  # F16
    2: (32, 18),  # Q4_0
    3: (32, 20),  # Q4_1
    6: (32, 22),  # Q5_0
    7: (32, 24),  # Q5_1
    8: (32, 34),  # Q8_0
    9: (32, 40),  # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),  # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),  # I8
    25: (1, 2),  # I16
    26: (1, 4),  # I32
    27: (1, 8),  # I64
    28: (1, 8),  # F64
    29: (256, 56),  # IQ1_M
    30: (1, 2),  # BF16
    34: (256, 54),  # TQ1_0
    35: (256, 66),  # TQ2_0
    39: (32, 17),  # MXFP4
    40: (64, 36),  # NVFP4
    41: (128, 18),  # Q1_0
}

TYPE_NAMES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
    35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0",
}


class TensorRole(IntEnum):
    """What a tensor does, which is what decides where it should live.

    The planner's whole job rests on one asymmetry: ROUTED_EXPERTS are enormous
    but only a fraction runs per token, so they tolerate living in slow memory.
    Everything else runs on every single token and belongs on the GPU.
    """

    TOKEN_EMBD = 0
    OUTPUT = 1
    NORM = 2
    ATTENTION = 3
    ROUTER = 4
    SHARED_EXPERTS = 5
    ROUTED_EXPERTS = 6
    FFN_DENSE = 7
    OTHER = 8
    SSM = 9
    """State-space (Mamba-style) layers. Hybrid models interleave these with
    attention; they run on every token just as attention does."""
    VISION = 10
    """Vision encoder. Idle unless the prompt contains an image."""
    MTP = 11
    """Multi-token prediction head, used only for speculative decoding."""
    AUDIO = 13
    """Audio encoder. Idle unless the prompt contains audio, like VISION."""
    PER_LAYER_EMBD = 14
    """Per-layer embedding table, as used by "effective parameter" designs.

    Large on disk and cold in practice: it is indexed by token id, so a decode
    step reads one row per layer rather than streaming the table. Charging it
    per token would make an 8.9 GiB model look three times slower than it is.
    """
    BLOCK_OTHER = 12
    """A per-block tensor this version does not recognise.

    Treated as hot, because it sits inside the per-layer forward path and runs
    on every token whatever it is called. The two ways of being wrong are not
    symmetric: guessing cold promises decode speed the machine cannot deliver,
    while guessing hot only spends VRAM budget that the planner then declines to
    fill with experts.
    """


#: 20260725 RG Roles that execute for every token of text generation.
HOT_ROLES = frozenset({
    TensorRole.TOKEN_EMBD,
    TensorRole.OUTPUT,
    TensorRole.NORM,
    TensorRole.ATTENTION,
    TensorRole.ROUTER,
    TensorRole.SHARED_EXPERTS,
    TensorRole.FFN_DENSE,
    TensorRole.SSM,
    TensorRole.BLOCK_OTHER,
})


class GGUFError(Exception):
    """Raised when a file is not valid GGUF or uses an unsupported version."""


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dimensions: tuple[int, ...]
    type_id: int
    offset: int
    """Byte offset relative to the start of the tensor data section."""

    @property
    def type_name(self) -> str:
        return TYPE_NAMES.get(self.type_id, f"UNKNOWN({self.type_id})")

    @property
    def n_elements(self) -> int:
        return prod(self.dimensions) if self.dimensions else 0

    @property
    def n_bytes(self) -> int:
        """Exact on-disk size of this tensor."""
        sizes = QUANT_SIZES.get(self.type_id)
        if sizes is None:
            raise GGUFError(f"unknown ggml type id {self.type_id} for tensor {self.name!r}")
        block_size, type_size = sizes
        elements = self.n_elements
        if elements % block_size:
            raise GGUFError(
                f"tensor {self.name!r} has {elements} elements, "
                f"not a multiple of block size {block_size}"
            )
        return elements // block_size * type_size

    @property
    def layer(self) -> int | None:
        """Block index for per-layer tensors, None for global ones."""
        if not self.name.startswith("blk."):
            return None
        parts = self.name.split(".", 2)
        if len(parts) < 2 or not parts[1].isdigit():
            return None
        return int(parts[1])

    @property
    def role(self) -> TensorRole:
        return classify(self.name)


def classify(name: str) -> TensorRole:
    """Map a tensor name to its role.

    Ordering matters throughout: `ffn_gate_inp` is the router and must not be
    mistaken for a dense gate, and `ffn_gate_shexp` (shared expert, runs every
    token) must not be mistaken for `ffn_gate_exps` (routed experts, mostly idle).

    Submodel prefixes are checked first, because a vision block's `attn_q` is
    spelled exactly like the language model's but is idle for text-only prompts.
    """
    if name.startswith("v."):
        return TensorRole.VISION
    if name.startswith("a."):
        return TensorRole.AUDIO
    if name.startswith("mtp."):
        return TensorRole.MTP
    # 20260725 RG Indexed per token, not streamed: gemma4:e4b keeps 5.25 of 8.9 GiB here.
    if name.startswith("per_layer_token_embd"):
        return TensorRole.PER_LAYER_EMBD

    stem = name.rsplit(".", 1)[0] if name.endswith((".weight", ".bias")) else name
    tail = stem.split(".")[-1] if stem.startswith("blk.") else stem

    if tail.startswith("ssm_"):
        return TensorRole.SSM

    if tail.endswith("_norm") or tail in {"output_norm", "attn_norm", "ffn_norm"}:
        return TensorRole.NORM
    if tail in {"token_embd", "pos_embd"}:
        return TensorRole.TOKEN_EMBD
    if tail == "output":
        return TensorRole.OUTPUT
    if tail.startswith("attn_"):
        return TensorRole.ATTENTION
    # 20260725 RG Router first: `ffn_gate_inp` picks the experts, it is not one.
    if tail in {"ffn_gate_inp", "ffn_gate_inp_shexp", "ffn_exp_probs_b"}:
        return TensorRole.ROUTER
    if tail.endswith("_shexp"):
        return TensorRole.SHARED_EXPERTS
    if tail.endswith("_exps") or tail.endswith("_exp"):
        return TensorRole.ROUTED_EXPERTS
    if tail.startswith("ffn_"):
        return TensorRole.FFN_DENSE
    # 20260725 RG An unknown tensor in a block still runs every token; cold cost 3x.
    if name.startswith("blk."):
        return TensorRole.BLOCK_OTHER
    return TensorRole.OTHER


@dataclass
class GGUFModel:
    path: Path | None
    version: int
    alignment: int
    data_offset: int
    """Absolute byte offset where the tensor data section begins."""
    metadata: dict[str, object]
    tensors: list[TensorInfo] = field(default_factory=list)
    big_endian: bool = False

    # -- metadata helpers -------------------------------------------------

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", "unknown"))

    def arch_key(self, suffix: str) -> object | None:
        """Look up an architecture-scoped key, e.g. arch_key('block_count')."""
        return self.metadata.get(f"{self.architecture}.{suffix}")

    @property
    def block_count(self) -> int | None:
        return _as_int(self.arch_key("block_count"))

    @property
    def embedding_length(self) -> int | None:
        return _as_int(self.arch_key("embedding_length"))

    @property
    def head_count(self) -> int | None:
        return _as_int(self.arch_key("attention.head_count"))

    @property
    def head_count_kv(self) -> int | None:
        """KV heads per attention layer.

        Hybrid SSM/attention models publish this as a per-layer array with zeros
        on the state-space layers, e.g. `[0, 0, 0, 4, ...]`. Reading that as a
        scalar fails, and silently falling back to `head_count` overstates the
        KV cache several times over. Take the largest declared value: that is
        what an actual attention layer uses.
        """
        raw = self.arch_key("attention.head_count_kv")
        if isinstance(raw, list):
            values = [value for value in (_as_int(item) for item in raw) if value]
            return max(values) if values else None
        return _as_int(raw) or self.head_count

    @property
    def attention_layer_count(self) -> int | None:
        """Layers that actually hold a KV cache.

        In a hybrid model most layers are state-space and carry a fixed-size
        state instead of a cache that grows with context, so charging every
        layer for KV would inflate the estimate by the ratio between them.
        """
        raw = self.arch_key("attention.head_count_kv")
        if isinstance(raw, list):
            return sum(1 for item in raw if _as_int(item))
        return self.block_count

    @property
    def head_dim(self) -> int | None:
        """Size of one attention head.

        Prefer the declared key length: deriving it as embedding/head_count
        assumes the heads tile the embedding exactly, which newer architectures
        no longer do (Qwen3.5 declares 256 where the division gives 213).
        """
        declared = _as_int(self.arch_key("attention.key_length"))
        if declared:
            return declared
        if self.embedding_length and self.head_count:
            return self.embedding_length // self.head_count
        return None

    @property
    def context_length(self) -> int | None:
        return _as_int(self.arch_key("context_length"))

    @property
    def expert_count(self) -> int:
        return _as_int(self.arch_key("expert_count")) or 0

    @property
    def expert_used_count(self) -> int:
        return _as_int(self.arch_key("expert_used_count")) or 0

    @property
    def is_moe(self) -> bool:
        return self.expert_count > 0

    # -- size accounting --------------------------------------------------

    @property
    def total_bytes(self) -> int:
        return sum(tensor.n_bytes for tensor in self.tensors)

    def bytes_by_role(self) -> dict[TensorRole, int]:
        totals: dict[TensorRole, int] = {}
        for tensor in self.tensors:
            totals[tensor.role] = totals.get(tensor.role, 0) + tensor.n_bytes
        return totals

    @property
    def hot_bytes(self) -> int:
        """Bytes touched on every token -- the GPU's first claim."""
        return sum(t.n_bytes for t in self.tensors if t.role in HOT_ROLES)

    @property
    def cold_bytes(self) -> int:
        """Everything not read on every token, and therefore offloadable.

        Routed experts are the bulk of this in a MoE, but a multimodal model
        also parks a vision encoder here, and a speculative-decoding head too.
        Together with `hot_bytes` this partitions the model exactly.
        """
        return sum(t.n_bytes for t in self.tensors if t.role not in HOT_ROLES)

    @property
    def routed_expert_bytes(self) -> int:
        """Bytes that are candidates for offloading to system RAM."""
        return sum(
            t.n_bytes for t in self.tensors if t.role is TensorRole.ROUTED_EXPERTS
        )

    def active_bytes_per_token(self) -> int:
        """Weight bytes read to produce one token.

        For a MoE this is far below total size: only `expert_used_count` of
        `expert_count` routed experts fire per token. This is the numerator of
        the decode-speed estimate, so it is the single most important number
        the planner computes.

        Approximate by construction -- it assumes experts are uniformly sized
        and ignores routing skew, where hot experts are read more often than
        cold ones.
        """
        active = self.hot_bytes
        if self.is_moe and self.expert_count:
            fraction = min(self.expert_used_count / self.expert_count, 1.0)
            active += int(self.routed_expert_bytes * fraction)
        else:
            active += self.routed_expert_bytes
        return active


def _as_int(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


class _Cursor:
    """Sequential reader that tracks absolute position for offset arithmetic."""

    def __init__(self, stream: BinaryIO, big_endian: bool = False) -> None:
        self._stream = stream
        self._prefix = ">" if big_endian else "<"

    @property
    def position(self) -> int:
        return self._stream.tell()

    def read(self, count: int) -> bytes:
        data = self._stream.read(count)
        if len(data) != count:
            raise GGUFError(f"unexpected end of file: wanted {count} bytes, got {len(data)}")
        return data

    def skip(self, count: int) -> None:
        self._stream.seek(count, 1)

    def scalar(self, fmt: str, size: int) -> object:
        return struct.unpack(self._prefix + fmt, self.read(size))[0]

    def u32(self) -> int:
        return int(self.scalar("I", 4))

    def u64(self) -> int:
        return int(self.scalar("Q", 8))

    def string(self) -> str:
        return self.read(self.u64()).decode("utf-8", errors="replace")

    def value(self, value_type: int) -> object:
        if value_type == ValueType.STRING:
            return self.string()
        if value_type == ValueType.ARRAY:
            return self.array()
        fmt = _SCALAR_FORMATS.get(value_type)
        if fmt is None:
            raise GGUFError(f"unknown metadata value type {value_type}")
        return self.scalar(*fmt)

    def array(self) -> object:
        element_type = self.u32()
        count = self.u64()

        if count > LARGE_ARRAY_THRESHOLD:
            if element_type == ValueType.STRING:
                for _ in range(count):
                    self.skip(self.u64())
            elif element_type == ValueType.ARRAY:
                for _ in range(count):
                    self.array()
            else:
                fmt = _SCALAR_FORMATS.get(element_type)
                if fmt is None:
                    raise GGUFError(f"unknown array element type {element_type}")
                self.skip(fmt[1] * count)
            return f"<{count} elements omitted>"

        return [self.value(element_type) for _ in range(count)]


def read(source: str | Path | BinaryIO) -> GGUFModel:
    """Parse the header of a GGUF file. Tensor data is never read."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        with path.open("rb") as stream:
            return _read_stream(stream, path)
    return _read_stream(source, None)


def _read_stream(stream: BinaryIO, path: Path | None) -> GGUFModel:
    magic = stream.read(4)
    if magic != GGUF_MAGIC:
        raise GGUFError(f"not a GGUF file: magic is {magic!r}, expected {GGUF_MAGIC!r}")

    # 20260725 RG GGUF v3 allows big-endian files.
    raw_version = stream.read(4)
    version = struct.unpack("<I", raw_version)[0]
    big_endian = version > 0xFFFF
    if big_endian:
        version = struct.unpack(">I", raw_version)[0]

    if version not in SUPPORTED_VERSIONS:
        raise GGUFError(f"unsupported GGUF version {version}, expected one of {SUPPORTED_VERSIONS}")

    cursor = _Cursor(stream, big_endian)
    tensor_count = cursor.u64()
    metadata_count = cursor.u64()

    metadata: dict[str, object] = {}
    for _ in range(metadata_count):
        key = cursor.string()
        metadata[key] = cursor.value(cursor.u32())

    tensors = []
    for _ in range(tensor_count):
        name = cursor.string()
        dimensions = tuple(cursor.u64() for _ in range(cursor.u32()))
        tensors.append(
            TensorInfo(
                name=name,
                dimensions=dimensions,
                type_id=cursor.u32(),
                offset=cursor.u64(),
            )
        )

    alignment = _as_int(metadata.get("general.alignment")) or DEFAULT_ALIGNMENT
    data_offset = _align(cursor.position, alignment)

    return GGUFModel(
        path=path,
        version=version,
        alignment=alignment,
        data_offset=data_offset,
        metadata=metadata,
        tensors=tensors,
        big_endian=big_endian,
    )


def _align(offset: int, alignment: int) -> int:
    return offset + (alignment - offset % alignment) % alignment
