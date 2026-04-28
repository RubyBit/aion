# Copyright (c) 2026 Angelos-Ermis Mangos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Shared AION v4 package writer + quantization helpers.

This module contains the canonical python-side encoder for the on-disk `.aion`
package format described by `src/aion/storage/aion_file/{types,parse,write}.zig`.
It is used by `convert_silero_vad_to_aion.py` and `convert_gemma4_e2b_to_aion.py`.

What's here:

- `Package`: a dataclass holding everything an `.aion` file contains (initializers,
  values, nodes, signatures, dim symbols/exprs, metadata, debug names, IO aliases).
- `write_aion_v4(path, pkg)`: serializes a `Package` to the v4 container format.
- `pack_q8_0(arr, axis)`: quantizes an fp32 numpy array to the ggml-compatible q8_0
  block layout that Aion's tiled storage consumes (see below).
- `Initializer.plain(...)` / `Initializer.quantized(...)` constructors and node/attr
  helpers.

Packed-quant convention (must match Aion's `TiledTensor` layout):
- `quant_axis` selects the block axis. Along that axis every 32 elements form one
  34-byte q8_0 block: `[f16 scale (little-endian, 2B)][32 × i8]`.
- Block-space shape replaces `shape[quant_axis]` with `shape[quant_axis] / 32`.
- Packed bytes are row-major over block-space, each element being one 34-byte block.

Do not add format changes here without updating `aion_file/write.zig` and
`aion_file/parse.zig` in lockstep.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------- AION v4 constants -----------------------------

MAGIC: bytes = b"AION"
VERSION: int = 4
HEADER_SIZE: int = 72
SECTION_DESC_SIZE: int = 24
INVALID_INDEX_U32: int = 0xFFFFFFFF


class SectionType:
    strings = 1
    tensors = 2
    values = 3
    nodes = 4
    signatures = 5
    graph_meta = 6
    dim_symbols = 7
    dim_exprs = 8
    metadata = 9
    debug_names = 10
    io_aliases = 12


SECTION_REQUIRED_FLAG: int = 1 << 0


class DType:
    # Keep in sync with src/aion/backend/types.zig.
    f32 = 0
    f16 = 1
    i8 = 2
    q4_0 = 3
    q8_0 = 4
    i32 = 5

    _SIZES: Dict[int, int] = {f32: 4, f16: 2, i8: 1, i32: 4}

    @classmethod
    def scalar_byte_size(cls, dt: int) -> int:
        if dt not in cls._SIZES:
            raise ValueError(f"scalar_byte_size: dtype {dt} is not scalar")
        return cls._SIZES[dt]


class ValueSource:
    public_input = 0
    initializer = 1
    produced = 2


class ElemwiseBinaryOp:
    add = 0
    sub = 1
    mul = 2
    div = 3


class UnaryOp:
    relu = 0
    gelu = 1
    silu = 2
    sigmoid = 3
    tanh = 4
    sqrt = 5


class ReduceOp:
    sum = 0
    mean = 1


class PadMode:
    zero = 0
    reflect = 1


class NodeKind:
    # Keep in sync with the NodeOpKind enum produced by `graph.OpTag` in
    # `src/aion/graph/graph.zig` (stable on-disk ordering via `write.zig`).
    MatMul = 0
    ElemwiseBinary = 1
    BroadcastLastDimBinary = 2
    Unary = 3
    Softmax = 4
    Conv1D = 5
    Conv2D = 6
    LayerNorm = 7
    RMSNorm = 8
    Attention = 9
    MultiHeadAttention = 10
    Reduce = 11
    Concat = 12
    LSTMCell = 13
    ComplexAbsMean = 14
    Copy = 15
    ViewReshape = 16
    ViewSqueeze = 17
    ViewUnsqueeze = 18
    ViewTranspose2D = 19
    ViewSliceND = 20
    GatherRows = 21
    RoPE1D = 22
    KVCacheAppend = 23
    MultiHeadAttentionCached = 24
    Cast = 25
    MatMulNT = 26


# q8_0 block constants (ggml-compatible).
Q8_0_BLOCK_ELEMS: int = 32
Q8_0_BLOCK_BYTES: int = 34  # 2B f16 scale + 32 × i8


# ------------------------------- byte helpers --------------------------------


def u8(x: int) -> bytes:
    return struct.pack("<B", x)


def u32(x: int) -> bytes:
    return struct.pack("<I", x)


def u64(x: int) -> bytes:
    return struct.pack("<Q", x)


def i32(x: int) -> bytes:
    return struct.pack("<i", x)


def f32_bytes(x: float) -> bytes:
    return struct.pack("<f", float(x))


def shape_term_constant(value: int) -> bytes:
    # kind u8 (0) + 7 padding + u64 payload
    return u8(0) + (b"\x00" * 7) + u64(int(value))


def shape_term_expr(expr_idx: int) -> bytes:
    return u8(1) + (b"\x00" * 7) + u64(int(expr_idx))


def shape_terms_for(shape: Sequence[int], batch_expr_index: Optional[int] = None) -> bytes:
    """Encode a shape as `u32 count + shape_terms`.

    If `batch_expr_index` is provided, the leading axis is emitted as an expr reference
    (for symbolic batch dims); all remaining axes are constants.
    """
    out = bytearray()
    out += u32(len(shape))
    for axis, dim in enumerate(shape):
        if axis == 0 and batch_expr_index is not None:
            out += shape_term_expr(batch_expr_index)
        else:
            out += shape_term_constant(int(dim))
    return bytes(out)


# ------------------------------ q8_0 packing ---------------------------------


def pack_q8_0(arr: np.ndarray, axis: int) -> bytes:
    """Quantize `arr` (any float dtype) into the q8_0 block layout Aion expects.

    Convention (matches `src/aion/storage/storage.zig`):
    - Every 32 consecutive elements along `axis` form one 34-byte block:
      `[little-endian f16 scale][32 × i8 quantized values]`.
    - Output bytes are row-major over the block-space shape (i.e., `arr.shape` with
      `arr.shape[axis]` replaced by `arr.shape[axis] // 32`).

    Callers are responsible for arranging the tensor so that quantizing along `axis`
    matches the semantics they want (e.g., per-row scales on an embedding `[V, D]`
    table → pass `axis=1` so each row owns its 48 scales).
    """
    if axis < 0:
        axis += arr.ndim
    if not (0 <= axis < arr.ndim):
        raise ValueError(f"pack_q8_0: axis {axis} out of range for rank {arr.ndim}")
    if arr.shape[axis] % Q8_0_BLOCK_ELEMS != 0:
        raise ValueError(
            f"pack_q8_0: axis {axis} must be a multiple of {Q8_0_BLOCK_ELEMS}, "
            f"got shape {arr.shape}"
        )

    # Move the quant axis to the end for simpler reshaping: shape = (..., A).
    x = np.moveaxis(arr.astype(np.float32, copy=False), axis, -1)
    lead_shape = x.shape[:-1]
    a = x.shape[-1]
    blocks_per_axis = a // Q8_0_BLOCK_ELEMS

    # Reshape to (..., blocks, 32).
    x_blk = x.reshape(*lead_shape, blocks_per_axis, Q8_0_BLOCK_ELEMS)

    absmax = np.max(np.abs(x_blk), axis=-1)  # (..., blocks)
    # Round-to-nearest q = round(x / scale) with scale = absmax/127. Zero absmax
    # blocks get scale=0 and quantize to all zeros (stored with inv_scale=0).
    scale_f32 = absmax / 127.0
    # Store scale in f16 so Zig-side dequant recovers the exact scale used at compute time.
    scale_f16 = scale_f32.astype(np.float16)
    # Recover the effective f32 scale after f16 rounding so the quantization step
    # matches what Aion sees on load.
    eff_scale = scale_f16.astype(np.float32)
    inv_scale = np.where(eff_scale != 0.0, 1.0 / np.maximum(eff_scale, 1e-30), 0.0)
    # Zero out inv_scale cleanly where eff_scale is 0.
    inv_scale = np.where(eff_scale == 0.0, 0.0, inv_scale)

    q = np.rint(x_blk * inv_scale[..., None]).clip(-128, 127).astype(np.int8)

    # Interleave per-block: 2 bytes scale (f16 little-endian) + 32 bytes i8.
    scale_bytes = np.ascontiguousarray(scale_f16).view(np.uint8).reshape(*lead_shape, blocks_per_axis, 2)
    q_bytes = np.ascontiguousarray(q).view(np.uint8)

    block_bytes = np.concatenate([scale_bytes, q_bytes], axis=-1)  # (..., blocks, 34)
    # Move the block axis back to the position matching block-space layout.
    # Current shape: (*lead_shape_moved, blocks, 34). `lead_shape` is the original
    # shape with axis `axis` moved to the end; to get block-space row-major bytes we
    # transpose blocks back into the original axis position.
    # Easy way: flatten everything except the trailing 34 into a single block dim,
    # written in the canonical block-space traversal order.
    #
    # Block-space shape is the original `arr.shape` with `arr.shape[axis]` replaced
    # by `blocks_per_axis`. Flattening that in C-order (row-major) matches Aion's
    # expected packed byte stream.
    block_shape = list(arr.shape)
    block_shape[axis] = blocks_per_axis

    # Reintroduce the block axis at its original position.
    # Current layout: `lead_shape + (blocks, 34)` where lead_shape = arr.shape with
    # axis removed. Reorder to `block_shape + (34,)` by inserting `blocks` back in.
    tmp = block_bytes.reshape(*lead_shape, blocks_per_axis, Q8_0_BLOCK_BYTES)
    # Build the permutation that moves the trailing block dim back to position `axis`.
    nd = tmp.ndim  # == arr.ndim + 1
    perm = list(range(nd))
    # Remove trailing block axis (index nd-2) and insert at `axis`.
    block_idx = nd - 2
    perm.remove(block_idx)
    perm.insert(axis, block_idx)
    tmp = np.ascontiguousarray(np.transpose(tmp, perm))

    expected_bytes = int(np.prod(block_shape)) * Q8_0_BLOCK_BYTES
    packed = tmp.tobytes(order="C")
    if len(packed) != expected_bytes:
        raise AssertionError(
            f"pack_q8_0: expected {expected_bytes} bytes but produced {len(packed)}"
        )
    return packed


# ----------------------------- package dataclasses ---------------------------


@dataclass
class Initializer:
    """Serialized tensor data for one initializer.

    Use `Initializer.plain(...)` or `Initializer.quantized(...)` rather than
    constructing directly to avoid invalid encodings.
    """

    kind: int  # 0=plain, 1=quantized
    plain_dtype: int  # meaningful for kind=0; also used as block_dtype for kind=1 (always 0 on disk)
    logical_dtype: int  # for kind=0 same as plain_dtype; for kind=1 the dequant-target scalar dtype (f16/f32)
    scheme: str  # kind=1 only ("q8_0" / "q4_0"); interned during write
    block_elems: int
    block_bytes: int
    quant_axis: int  # kind=1: block axis; kind=0: 0
    params: bytes
    data: bytes

    @classmethod
    def plain(cls, dtype: int, data: bytes) -> "Initializer":
        size = DType.scalar_byte_size(dtype)
        return cls(
            kind=0,
            plain_dtype=dtype,
            logical_dtype=dtype,
            scheme="",
            block_elems=1,
            block_bytes=size,
            quant_axis=0,
            params=b"",
            data=data,
        )

    @classmethod
    def quantized_q8_0(cls, logical_dtype: int, quant_axis: int, data: bytes) -> "Initializer":
        if logical_dtype not in (DType.f16, DType.f32):
            raise ValueError(
                f"quantized_q8_0: logical_dtype must be f16 or f32 (got {logical_dtype})"
            )
        return cls(
            kind=1,
            plain_dtype=0,  # unused for kind=1 on disk
            logical_dtype=logical_dtype,
            scheme="q8_0",
            block_elems=Q8_0_BLOCK_ELEMS,
            block_bytes=Q8_0_BLOCK_BYTES,
            quant_axis=quant_axis,
            params=b"",
            data=data,
        )


@dataclass
class ValueRecord:
    dtype: int
    rank: int
    source: int
    shape_terms: bytes  # pre-encoded: u32 count + shape_terms
    initializer_index: Optional[int] = None


@dataclass
class NodeRecord:
    kind: int
    output: int
    inputs: List[int]
    attr: bytes


@dataclass
class NamedValue:
    name: str
    value: int


@dataclass
class DebugName:
    value: int
    name: str


@dataclass
class MetadataEntry:
    key: str
    value: str


@dataclass
class IoAlias:
    input_index: int
    output_index: int


@dataclass
class Package:
    initializers: List[Initializer] = field(default_factory=list)
    values: List[ValueRecord] = field(default_factory=list)
    nodes: List[NodeRecord] = field(default_factory=list)
    inputs: List[NamedValue] = field(default_factory=list)
    outputs: List[NamedValue] = field(default_factory=list)
    dim_symbols: List[str] = field(default_factory=list)
    # Each entry is a symbol index; only `.symbol(sym)` exprs are emitted (arithmetic
    # exprs can be added here later).
    dim_exprs_symbol_indices: List[int] = field(default_factory=list)
    metadata: List[MetadataEntry] = field(default_factory=list)
    debug_names: List[DebugName] = field(default_factory=list)
    io_aliases: List[IoAlias] = field(default_factory=list)


# ------------------------------- node attrs ---------------------------------
#
# Mirror `src/aion/storage/aion_file/write.zig:encodeNodeOp` byte-for-byte.


def attr_matmul(alpha: float = 1.0, beta: float = 0.0) -> bytes:
    return f32_bytes(alpha) + f32_bytes(beta)


def attr_unary(op: int) -> bytes:
    return u8(op)


def attr_binary(op: int) -> bytes:
    return u8(op)


def attr_softmax(axis: int) -> bytes:
    return i32(axis)


def attr_conv1d(
    stride: int, dilation: int, pad_left: int, pad_right: int, pad_mode: int, groups: int
) -> bytes:
    return (
        u64(stride)
        + u64(dilation)
        + u64(pad_left)
        + u64(pad_right)
        + u8(pad_mode)
        + u64(groups)
    )


def attr_layernorm(eps: float, normalized_shape_terms: Sequence[bytes]) -> bytes:
    out = bytearray()
    out += f32_bytes(eps)
    out += u32(len(normalized_shape_terms))
    for t in normalized_shape_terms:
        out += t
    return bytes(out)


def attr_rmsnorm(eps: float, normalized_shape_terms: Sequence[bytes]) -> bytes:
    return attr_layernorm(eps, normalized_shape_terms)


def attr_attention(scale: float, causal: bool) -> bytes:
    return f32_bytes(scale) + u8(1 if causal else 0)


def attr_mha(scale: float, causal: bool, heads: int) -> bytes:
    return f32_bytes(scale) + u8(1 if causal else 0) + u64(heads)


def attr_mha_cached(
    scale: float, causal: bool, sliding_window: int, attn_logits_soft_cap: float
) -> bytes:
    return (
        f32_bytes(scale)
        + u8(1 if causal else 0)
        + u64(sliding_window)
        + f32_bytes(attn_logits_soft_cap)
    )


def attr_reduce(op: int, axis: Optional[int]) -> bytes:
    out = bytearray()
    out += u8(op)
    out += u8(1 if axis is not None else 0)
    if axis is not None:
        out += i32(int(axis))
    return bytes(out)


def attr_concat(axis: int) -> bytes:
    return i32(axis)


def attr_lstm(has_bias: bool) -> bytes:
    return u8(1 if has_bias else 0)


def attr_complex_abs_mean(out_channels: int) -> bytes:
    return u64(out_channels)


def attr_view_reshape(new_shape_terms: Sequence[bytes]) -> bytes:
    out = bytearray()
    out += u32(len(new_shape_terms))
    for t in new_shape_terms:
        out += t
    return bytes(out)


def attr_view_squeeze(axis: Optional[int]) -> bytes:
    out = bytearray()
    out += u8(1 if axis is not None else 0)
    if axis is not None:
        out += i32(int(axis))
    return bytes(out)


def attr_view_unsqueeze(axis: int) -> bytes:
    return i32(axis)


def attr_view_slice_nd(starts: Sequence[int], lens_terms: Sequence[bytes]) -> bytes:
    if len(starts) != len(lens_terms):
        raise ValueError("attr_view_slice_nd: starts and lens must be same length")
    out = bytearray()
    out += u32(len(starts))
    for s in starts:
        out += u64(int(s))
    out += u32(len(lens_terms))
    for t in lens_terms:
        out += t
    return bytes(out)


def attr_rope1d(base_frequency: float, scale_factor: float, rope_proportion: float) -> bytes:
    return f32_bytes(base_frequency) + f32_bytes(scale_factor) + f32_bytes(rope_proportion)


def attr_cast(to_dtype: int) -> bytes:
    return u8(to_dtype)


def attr_matmul_nt(alpha: float = 1.0, beta: float = 0.0) -> bytes:
    return f32_bytes(alpha) + f32_bytes(beta)


# ------------------------------- file writer --------------------------------


class _StringInterner:
    def __init__(self) -> None:
        self._map: Dict[bytes, int] = {}
        self._items: List[bytes] = []

    def put(self, s: str) -> int:
        b = s.encode("utf-8")
        if b in self._map:
            return self._map[b]
        idx = len(self._items)
        self._items.append(b)
        self._map[b] = idx
        return idx

    def get(self, s: str) -> int:
        b = s.encode("utf-8")
        return self._map[b]

    @property
    def items(self) -> List[bytes]:
        return self._items


def _encode_strings(interner: _StringInterner) -> bytes:
    out = bytearray()
    out += u32(len(interner.items))
    for s in interner.items:
        out += u32(len(s))
        out += s
    return bytes(out)


def _encode_dim_symbols(interner: _StringInterner, dim_symbols: List[str]) -> bytes:
    out = bytearray()
    out += u32(len(dim_symbols))
    for s in dim_symbols:
        out += u32(interner.get(s))
    return bytes(out)


def _encode_dim_exprs(dim_exprs_symbol_indices: List[int]) -> bytes:
    # Each expr stored as `.symbol(idx)`. Matches `encodeDimExprsSection` in write.zig.
    out = bytearray()
    out += u32(len(dim_exprs_symbol_indices))
    for sym_idx in dim_exprs_symbol_indices:
        out += u8(0) + (b"\x00" * 3)
        out += u32(int(sym_idx))
        out += shape_term_constant(1)
        out += shape_term_constant(1)
    return bytes(out)


def _encode_initializers(interner: _StringInterner, initializers: List[Initializer]) -> bytes:
    out = bytearray()
    out += u32(len(initializers))
    for init in initializers:
        if init.kind == 0:
            scheme_idx = INVALID_INDEX_U32
        elif init.kind == 1:
            if not init.scheme:
                raise ValueError("quantized initializer missing scheme string")
            scheme_idx = interner.get(init.scheme)
        else:
            raise ValueError(f"unknown initializer kind: {init.kind}")

        out += u8(init.kind)
        out += u8(init.plain_dtype)
        out += u8(init.logical_dtype)
        out += u8(0)  # reserved
        out += u32(scheme_idx)
        out += u32(init.block_elems)
        out += u32(init.block_bytes)
        out += i32(init.quant_axis)
        out += u32(len(init.params))
        out += u64(len(init.data))
        out += init.params
        out += init.data
    return bytes(out)


def _encode_values(values: List[ValueRecord]) -> bytes:
    out = bytearray()
    out += u32(len(values))
    for v in values:
        out += u8(int(v.dtype))
        out += u8(int(v.rank))
        out += u8(int(v.source))
        out += u8(0)  # reserved
        out += u32(int(v.initializer_index) if v.initializer_index is not None else INVALID_INDEX_U32)
        out += v.shape_terms  # starts with u32 count
    return bytes(out)


def _encode_nodes(nodes: List[NodeRecord]) -> bytes:
    out = bytearray()
    out += u32(len(nodes))
    for n in nodes:
        out += u8(int(n.kind))
        out += b"\x00\x00\x00"  # padding
        out += u32(int(n.output))
        out += u32(len(n.inputs))
        out += u32(len(n.attr))
        for vi in n.inputs:
            out += u32(int(vi))
        out += n.attr
    return bytes(out)


def _encode_signatures(
    interner: _StringInterner,
    inputs: List[NamedValue],
    outputs: List[NamedValue],
) -> bytes:
    out = bytearray()
    out += u32(len(inputs))
    out += u32(len(outputs))
    for s in inputs:
        out += u32(interner.get(s.name))
        out += u32(int(s.value))
    for s in outputs:
        out += u32(interner.get(s.name))
        out += u32(int(s.value))
    return bytes(out)


def _encode_metadata(interner: _StringInterner, metadata: List[MetadataEntry]) -> bytes:
    out = bytearray()
    out += u32(len(metadata))
    for m in metadata:
        out += u32(interner.get(m.key))
        out += u32(interner.get(m.value))
    return bytes(out)


def _encode_debug_names(interner: _StringInterner, debug_names: List[DebugName]) -> bytes:
    out = bytearray()
    out += u32(len(debug_names))
    for dn in debug_names:
        out += u32(int(dn.value))
        out += u32(interner.get(dn.name))
    return bytes(out)


def _encode_io_aliases(io_aliases: List[IoAlias]) -> bytes:
    out = bytearray()
    out += u32(len(io_aliases))
    for a in io_aliases:
        out += u32(int(a.input_index))
        out += u32(int(a.output_index))
    return bytes(out)


def _encode_graph_meta(pkg: Package) -> bytes:
    out = bytearray()
    out += u32(len(pkg.values))
    out += u32(len(pkg.nodes))
    out += u32(len(pkg.initializers))
    out += u32(len(pkg.inputs))
    out += u32(len(pkg.outputs))
    out += u32(len(pkg.dim_symbols))
    out += u32(len(pkg.dim_exprs_symbol_indices))
    out += u32(len(pkg.metadata))
    out += u32(len(pkg.debug_names))
    out += u32(len(pkg.io_aliases))
    return bytes(out)


def _collect_strings(pkg: Package) -> _StringInterner:
    interner = _StringInterner()
    for init in pkg.initializers:
        if init.kind == 1:
            interner.put(init.scheme)
    for s in [nv.name for nv in pkg.inputs] + [nv.name for nv in pkg.outputs]:
        interner.put(s)
    for sym in pkg.dim_symbols:
        interner.put(sym)
    for m in pkg.metadata:
        interner.put(m.key)
        interner.put(m.value)
    for dn in pkg.debug_names:
        interner.put(dn.name)
    return interner


def write_aion_v4(path: str, pkg: Package) -> None:
    """Serialize `pkg` to `path` as a v4 `.aion` package."""
    interner = _collect_strings(pkg)

    sections: List[Tuple[int, int, bytes]] = []

    sections.append((SectionType.strings, SECTION_REQUIRED_FLAG, _encode_strings(interner)))
    sections.append((SectionType.tensors, SECTION_REQUIRED_FLAG, _encode_initializers(interner, pkg.initializers)))
    sections.append((SectionType.values, SECTION_REQUIRED_FLAG, _encode_values(pkg.values)))
    sections.append((SectionType.nodes, SECTION_REQUIRED_FLAG, _encode_nodes(pkg.nodes)))
    sections.append((SectionType.signatures, SECTION_REQUIRED_FLAG, _encode_signatures(interner, pkg.inputs, pkg.outputs)))
    sections.append((SectionType.graph_meta, SECTION_REQUIRED_FLAG, _encode_graph_meta(pkg)))

    if pkg.dim_symbols:
        sections.append((SectionType.dim_symbols, 0, _encode_dim_symbols(interner, pkg.dim_symbols)))
    if pkg.dim_exprs_symbol_indices:
        sections.append((SectionType.dim_exprs, 0, _encode_dim_exprs(pkg.dim_exprs_symbol_indices)))
    if pkg.metadata:
        sections.append((SectionType.metadata, 0, _encode_metadata(interner, pkg.metadata)))
    if pkg.debug_names:
        sections.append((SectionType.debug_names, 0, _encode_debug_names(interner, pkg.debug_names)))
    if pkg.io_aliases:
        sections.append((SectionType.io_aliases, 0, _encode_io_aliases(pkg.io_aliases)))

    dir_offset = HEADER_SIZE
    dir_size = len(sections) * SECTION_DESC_SIZE
    payload_offset = dir_offset + dir_size
    file_size = payload_offset + sum(len(s[2]) for s in sections)

    directory = bytearray()
    next_off = payload_offset
    for sec_type, flags, sec_bytes in sections:
        directory += u32(sec_type)
        directory += u32(flags)
        directory += u64(next_off)
        directory += u64(len(sec_bytes))
        next_off += len(sec_bytes)

    header = bytearray()
    header += MAGIC
    header += u32(VERSION)
    header += u32(len(sections))
    header += u32(0)  # reserved_u32
    header += u64(dir_offset)
    header += u64(file_size)
    header += u64(0)  # flags
    header += b"\x00" * 32

    assert len(header) == HEADER_SIZE, f"header size mismatch: {len(header)} != {HEADER_SIZE}"
    assert len(directory) == dir_size, f"directory size mismatch: {len(directory)} != {dir_size}"

    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(directory)
        for _, _, sec_bytes in sections:
            fh.write(sec_bytes)
