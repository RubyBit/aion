#!/usr/bin/env python3
"""Convert Silero VAD safetensors -> AION v4 (.aion) package.

This writes the **full** model package (weights + compute graph) in the exact
container format described in docs/AION_FORMAT.md and implemented by:
  src/aion/storage/aion_file/{write,parse}.zig

It targets the architecture used by `examples/silero_vad.zig`:
- Conv1D is **NLC** (channel-last) and Aion-native weight layout is [k, cin, cout]
- LSTMCell expects w_ih: [input, 4h] and w_hh: [hidden, 4h]

Usage:
  python scripts/convert_silero_vad_to_aion.py silero_vad_16k.safetensors silero_vad.aion \
    --num-samples 512 --context 64 --pad-mode reflect

Dependencies:
  pip install safetensors numpy

Notes:
- The exported package supports a symbolic batch dimension named "batch".
- Weight dtypes are exported as f32.
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _load_safetensors_numpy(path: str) -> Dict[str, np.ndarray]:
    try:
        from safetensors.numpy import load_file  # type: ignore

        return load_file(path)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency. Install with: pip install safetensors numpy\n"
            f"Import error: {e}"
        )


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
    # src/aion/backend/types.zig
    f32 = 0
    f16 = 1
    i8 = 2
    q4_0 = 3
    q8_0 = 4


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


# Node op-kind tags (src/aion/storage/aion_file/write.zig)
class NodeKind:
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


# ------------------------------- AION writers --------------------------------


def _u8(x: int) -> bytes:
    return struct.pack("<B", x)


def _u32(x: int) -> bytes:
    return struct.pack("<I", x)


def _u64(x: int) -> bytes:
    return struct.pack("<Q", x)


def _i32(x: int) -> bytes:
    return struct.pack("<i", x)


def _f32(x: float) -> bytes:
    return struct.pack("<f", float(x))


def _shape_term_constant(value: int) -> bytes:
    # kind u8 (0) + 7 padding + u64 payload
    return _u8(0) + (b"\x00" * 7) + _u64(int(value))


def _shape_term_expr(expr_idx: int) -> bytes:
    return _u8(1) + (b"\x00" * 7) + _u64(int(expr_idx))


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


@dataclass
class Initializer:
    dtype: int  # plain dtype only for now (f32)
    data: bytes


@dataclass
class ValueRecord:
    dtype: int
    rank: int
    source: int
    shape_terms: bytes  # encoded shape terms array (u32 count + terms...)
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
    # v4 sections
    initializers: List[Initializer]
    values: List[ValueRecord]
    nodes: List[NodeRecord]
    inputs: List[NamedValue]
    outputs: List[NamedValue]
    dim_symbols: List[str]
    dim_exprs_symbol_indices: List[int]  # each expr is a .symbol(sym_idx)
    metadata: List[MetadataEntry]
    debug_names: List[DebugName]
    io_aliases: List[IoAlias]


def _encode_strings_section(interner: _StringInterner) -> bytes:
    out = bytearray()
    out += _u32(len(interner.items))
    for s in interner.items:
        out += _u32(len(s))
        out += s
    return bytes(out)


def _encode_dim_symbols_section(interner: _StringInterner, dim_symbols: List[str]) -> bytes:
    out = bytearray()
    out += _u32(len(dim_symbols))
    for s in dim_symbols:
        out += _u32(interner.get(s))
    return bytes(out)


def _encode_dim_exprs_section(dim_exprs_symbol_indices: List[int]) -> bytes:
    # Mirrors src/aion/storage/aion_file/write.zig for DimExpr.symbol:
    # kind=0, 3 pad, aux=u32 sym_idx, then two ShapeTerms (both constant 1).
    out = bytearray()
    out += _u32(len(dim_exprs_symbol_indices))
    for sym_idx in dim_exprs_symbol_indices:
        out += _u8(0) + (b"\x00" * 3)
        out += _u32(int(sym_idx))
        out += _shape_term_constant(1)
        out += _shape_term_constant(1)
    return bytes(out)


def _encode_initializers_section(initializers: List[Initializer]) -> bytes:
    # Plain-only encoding (kind=0)
    out = bytearray()
    out += _u32(len(initializers))
    for init in initializers:
        dtype = int(init.dtype)
        out += _u8(0)  # kind=plain
        out += _u8(dtype)  # plain_dtype
        out += _u8(dtype)  # logical_dtype (same for plain)
        out += _u8(0)  # reserved
        out += _u32(INVALID_INDEX_U32)  # scheme string index
        out += _u32(1)  # block_elems for f32
        out += _u32(4)  # block_bytes for f32
        out += _i32(0)  # quant_axis
        out += _u32(0)  # params_len
        out += _u64(len(init.data))
        out += init.data
    return bytes(out)


def _encode_values_section(values: List[ValueRecord]) -> bytes:
    out = bytearray()
    out += _u32(len(values))
    for v in values:
        out += _u8(int(v.dtype))
        out += _u8(int(v.rank))
        out += _u8(int(v.source))
        out += _u8(0)  # reserved
        out += _u32(int(v.initializer_index) if v.initializer_index is not None else INVALID_INDEX_U32)
        # shape_terms is already encoded (u32 count + terms...)
        # But values section expects u32 shape_term_count then terms.
        # So we store shape_terms as the entire payload and just append it.
        # First 4 bytes are count.
        out += v.shape_terms
    return bytes(out)


def _encode_nodes_section(nodes: List[NodeRecord]) -> bytes:
    out = bytearray()
    out += _u32(len(nodes))
    for n in nodes:
        out += _u8(int(n.kind))
        out += b"\x00\x00\x00"  # padding
        out += _u32(int(n.output))
        out += _u32(len(n.inputs))
        out += _u32(len(n.attr))
        for vi in n.inputs:
            out += _u32(int(vi))
        out += n.attr
    return bytes(out)


def _encode_signatures_section(interner: _StringInterner, inputs: List[NamedValue], outputs: List[NamedValue]) -> bytes:
    out = bytearray()
    out += _u32(len(inputs))
    out += _u32(len(outputs))
    for s in inputs:
        out += _u32(interner.get(s.name))
        out += _u32(int(s.value))
    for s in outputs:
        out += _u32(interner.get(s.name))
        out += _u32(int(s.value))
    return bytes(out)


def _encode_metadata_section(interner: _StringInterner, metadata: List[MetadataEntry]) -> bytes:
    out = bytearray()
    out += _u32(len(metadata))
    for m in metadata:
        out += _u32(interner.get(m.key))
        out += _u32(interner.get(m.value))
    return bytes(out)


def _encode_debug_names_section(interner: _StringInterner, debug_names: List[DebugName]) -> bytes:
    out = bytearray()
    out += _u32(len(debug_names))
    for dn in debug_names:
        out += _u32(int(dn.value))
        out += _u32(interner.get(dn.name))
    return bytes(out)


def _encode_io_aliases_section(io_aliases: List[IoAlias]) -> bytes:
    out = bytearray()
    out += _u32(len(io_aliases))
    for a in io_aliases:
        out += _u32(int(a.input_index))
        out += _u32(int(a.output_index))
    return bytes(out)


def _encode_graph_meta_section(pkg: Package) -> bytes:
    out = bytearray()
    out += _u32(len(pkg.values))
    out += _u32(len(pkg.nodes))
    out += _u32(len(pkg.initializers))
    out += _u32(len(pkg.inputs))
    out += _u32(len(pkg.outputs))
    out += _u32(len(pkg.dim_symbols))
    out += _u32(len(pkg.dim_exprs_symbol_indices))
    out += _u32(len(pkg.metadata))
    out += _u32(len(pkg.debug_names))
    out += _u32(len(pkg.io_aliases))
    return bytes(out)


def write_aion_v4(path: str, pkg: Package) -> None:
    # Collect strings exactly like Zig: initializers(quant scheme), signatures, dim symbols, metadata, debug names.
    interner = _StringInterner()
    for s in [nv.name for nv in pkg.inputs] + [nv.name for nv in pkg.outputs]:
        interner.put(s)
    for sym in pkg.dim_symbols:
        interner.put(sym)
    for m in pkg.metadata:
        interner.put(m.key)
        interner.put(m.value)
    for dn in pkg.debug_names:
        interner.put(dn.name)

    sections: List[Tuple[int, int, bytes]] = []  # (type, flags, bytes)

    sections.append((SectionType.strings, SECTION_REQUIRED_FLAG, _encode_strings_section(interner)))
    sections.append((SectionType.tensors, SECTION_REQUIRED_FLAG, _encode_initializers_section(pkg.initializers)))
    sections.append((SectionType.values, SECTION_REQUIRED_FLAG, _encode_values_section(pkg.values)))
    sections.append((SectionType.nodes, SECTION_REQUIRED_FLAG, _encode_nodes_section(pkg.nodes)))
    sections.append((SectionType.signatures, SECTION_REQUIRED_FLAG, _encode_signatures_section(interner, pkg.inputs, pkg.outputs)))
    sections.append((SectionType.graph_meta, SECTION_REQUIRED_FLAG, _encode_graph_meta_section(pkg)))

    if pkg.dim_symbols:
        sections.append((SectionType.dim_symbols, 0, _encode_dim_symbols_section(interner, pkg.dim_symbols)))
    if pkg.dim_exprs_symbol_indices:
        sections.append((SectionType.dim_exprs, 0, _encode_dim_exprs_section(pkg.dim_exprs_symbol_indices)))
    if pkg.metadata:
        sections.append((SectionType.metadata, 0, _encode_metadata_section(interner, pkg.metadata)))
    if pkg.debug_names:
        sections.append((SectionType.debug_names, 0, _encode_debug_names_section(interner, pkg.debug_names)))
    if pkg.io_aliases:
        sections.append((SectionType.io_aliases, 0, _encode_io_aliases_section(pkg.io_aliases)))

    # Header + directory
    dir_offset = HEADER_SIZE
    dir_size = len(sections) * SECTION_DESC_SIZE
    payload_offset = dir_offset + dir_size
    file_size = payload_offset + sum(len(s[2]) for s in sections)

    # Build directory entries
    directory = bytearray()
    next_off = payload_offset
    for sec_type, flags, sec_bytes in sections:
        directory += _u32(sec_type)
        directory += _u32(flags)
        directory += _u64(next_off)
        directory += _u64(len(sec_bytes))
        next_off += len(sec_bytes)

    header = bytearray()
    header += MAGIC
    header += _u32(VERSION)
    header += _u32(len(sections))
    header += _u32(0)  # reserved_u32
    header += _u64(dir_offset)
    header += _u64(file_size)
    header += _u64(0)  # flags
    header += b"\x00" * 32

    assert len(header) == HEADER_SIZE
    assert len(directory) == dir_size

    with open(path, "wb") as f:
        f.write(header)
        f.write(directory)
        for _, _, sec_bytes in sections:
            f.write(sec_bytes)

    print(f"wrote {path} ({file_size} bytes, {len(sections)} sections)")


# ------------------------- Silero weight extraction --------------------------


@dataclass(frozen=True)
class ParamSpec:
    expected_shape: Tuple[int, ...]
    candidates: Tuple[str, ...]


def _format_shape(a: np.ndarray) -> str:
    return "(" + ", ".join(str(int(x)) for x in a.shape) + ")"


def _find_key(state: Dict[str, np.ndarray], candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in state:
            return c

    keys = list(state.keys())

    for c in candidates:
        m = [k for k in keys if k.endswith(c)]
        if len(m) == 1:
            return m[0]

    for c in candidates:
        m = [k for k in keys if c in k]
        if len(m) == 1:
            return m[0]

    return None


def _as_f32(a: np.ndarray) -> np.ndarray:
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    return np.ascontiguousarray(a)


def _conv_w_to_aion(a: np.ndarray, expected: Tuple[int, int, int]) -> np.ndarray:
    # Aion: [k, cin, cout]
    a = _as_f32(a)
    if a.ndim != 3:
        raise ValueError(f"conv weight must be rank-3, got {_format_shape(a)}")

    k, cin, cout = expected

    if tuple(int(x) for x in a.shape) == expected:
        return np.ascontiguousarray(a)

    # Torch: [cout, cin, k]
    if tuple(int(x) for x in a.shape) == (cout, cin, k):
        return np.ascontiguousarray(a.transpose(2, 1, 0))

    raise ValueError(f"unexpected conv weight shape for expected {expected}: got {_format_shape(a)}")


def _lstm_w_to_aion(a: np.ndarray, expected: Tuple[int, int]) -> np.ndarray:
    # Aion: [in_or_hidden, 4h]
    a = _as_f32(a)
    if a.ndim != 2:
        raise ValueError(f"lstm weight must be rank-2, got {_format_shape(a)}")

    if tuple(int(x) for x in a.shape) == expected:
        return np.ascontiguousarray(a)

    # Torch: [4h, in_or_hidden]
    if tuple(int(x) for x in a.shape) == (expected[1], expected[0]):
        return np.ascontiguousarray(a.transpose(1, 0))

    raise ValueError(f"unexpected lstm weight shape for expected {expected}: got {_format_shape(a)}")


def _bias_to_aion(a: np.ndarray, expected: Tuple[int, ...]) -> np.ndarray:
    a = _as_f32(a)
    if tuple(int(x) for x in a.shape) != expected:
        raise ValueError(f"unexpected bias shape for expected {expected}: got {_format_shape(a)}")
    return np.ascontiguousarray(a)


SPECS: Dict[str, ParamSpec] = {
    "stft_w": ParamSpec((256, 1, 258), ("stft_conv.weight", "stft.weight", "stft_conv/weight", "stft_conv_weight")),
    "conv1_w": ParamSpec((3, 129, 128), ("conv1.weight", "conv1/weight", "conv1.w")),
    "conv1_b": ParamSpec((128,), ("conv1.bias", "conv1/bias", "conv1.b")),
    "conv2_w": ParamSpec((3, 128, 64), ("conv2.weight", "conv2/weight", "conv2.w")),
    "conv2_b": ParamSpec((64,), ("conv2.bias", "conv2/bias", "conv2.b")),
    "conv3_w": ParamSpec((3, 64, 64), ("conv3.weight", "conv3/weight", "conv3.w")),
    "conv3_b": ParamSpec((64,), ("conv3.bias", "conv3/bias", "conv3.b")),
    "conv4_w": ParamSpec((3, 64, 128), ("conv4.weight", "conv4/weight", "conv4.w")),
    "conv4_b": ParamSpec((128,), ("conv4.bias", "conv4/bias", "conv4.b")),
    "lstm_w_ih": ParamSpec((128, 512), ("lstm_cell.weight_ih", "weight_ih_l0", "weight_ih")),
    "lstm_w_hh": ParamSpec((128, 512), ("lstm_cell.weight_hh", "weight_hh_l0", "weight_hh")),
    "lstm_b_ih": ParamSpec((512,), ("lstm_cell.bias_ih", "bias_ih_l0", "bias_ih")),
    "lstm_b_hh": ParamSpec((512,), ("lstm_cell.bias_hh", "bias_hh_l0", "bias_hh")),
    "final_w": ParamSpec((1, 128, 1), ("final_conv.weight", "final/weight", "final.weight")),
    "final_b": ParamSpec((1,), ("final_conv.bias", "final/bias", "final.bias")),
}


WEIGHT_ORDER: Tuple[str, ...] = (
    "stft_w",
    "conv1_w",
    "conv1_b",
    "conv2_w",
    "conv2_b",
    "conv3_w",
    "conv3_b",
    "conv4_w",
    "conv4_b",
    "lstm_w_ih",
    "lstm_w_hh",
    "lstm_b_ih",
    "lstm_b_hh",
    "final_w",
    "final_b",
)


def extract_weights(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name in WEIGHT_ORDER:
        spec = SPECS[name]
        key = _find_key(state, spec.candidates)
        if key is None:
            keys = "\n".join(f"  {k}: {state[k].dtype} {state[k].shape}" for k in sorted(state.keys()))
            raise KeyError(f"missing '{name}'. Tried {list(spec.candidates)}\n\nAvailable keys:\n{keys}")
        arr = state[key]

        if name.endswith("_w") and name not in ("lstm_w_ih", "lstm_w_hh"):
            out[name] = _conv_w_to_aion(arr, spec.expected_shape)  # type: ignore[arg-type]
        elif name in ("lstm_w_ih", "lstm_w_hh"):
            out[name] = _lstm_w_to_aion(arr, spec.expected_shape)  # type: ignore[arg-type]
        elif name.endswith("_b") or name == "final_b":
            out[name] = _bias_to_aion(arr, spec.expected_shape)
        else:
            out[name] = _as_f32(arr)

        if tuple(int(x) for x in out[name].shape) != spec.expected_shape:
            raise ValueError(f"bad shape for {name}: expected {spec.expected_shape}, got {_format_shape(out[name])} (key={key})")

    return out


def _shape_terms_for(shape: Sequence[int], batch_expr_index: Optional[int] = None) -> bytes:
    out = bytearray()
    out += _u32(len(shape))
    for axis, dim in enumerate(shape):
        if axis == 0 and batch_expr_index is not None:
            out += _shape_term_expr(batch_expr_index)
        else:
            out += _shape_term_constant(int(dim))
    return bytes(out)


def _node_attr_conv1d(stride: int, dilation: int, pad_left: int, pad_right: int, pad_mode: int, groups: int) -> bytes:
    return (
        _u64(stride)
        + _u64(dilation)
        + _u64(pad_left)
        + _u64(pad_right)
        + _u8(pad_mode)
        + _u64(groups)
    )


def _node_attr_unary(op: int) -> bytes:
    return _u8(op)


def _node_attr_binary(op: int) -> bytes:
    return _u8(op)


def _node_attr_lstm(has_bias: bool) -> bytes:
    return _u8(1 if has_bias else 0)


def _node_attr_reduce(op: int, axis: Optional[int]) -> bytes:
    out = bytearray()
    out += _u8(op)
    out += _u8(1 if axis is not None else 0)
    if axis is not None:
        out += _i32(int(axis))
    return bytes(out)


def _node_attr_squeeze(axis: Optional[int]) -> bytes:
    out = bytearray()
    out += _u8(1 if axis is not None else 0)
    if axis is not None:
        out += _i32(int(axis))
    return bytes(out)


def _node_attr_unsqueeze(axis: int) -> bytes:
    return _i32(int(axis))


def _node_attr_slice(starts: Sequence[int], lens_terms: Sequence[bytes]) -> bytes:
    if len(starts) != len(lens_terms):
        raise ValueError("starts and lens must have same length")
    out = bytearray()
    out += _u32(len(starts))
    for s in starts:
        out += _u64(int(s))
    out += _u32(len(lens_terms))
    for t in lens_terms:
        out += t
    return bytes(out)


# ----------------------------- Graph construction ----------------------------


def build_tiny_silero_aion_package(
    weights: Dict[str, np.ndarray],
    num_samples: int,
    context_size: int,
    pad_mode: int,
) -> Package:
    # Model constants (match examples/silero_vad.zig)
    n_fft = 256
    stride = 128
    pad = 64
    cutoff = (n_fft // 2) + 1

    chunk_input_len = int(num_samples + context_size)
    if chunk_input_len < n_fft:
        raise ValueError("chunk_input_len must be >= 256")

    padded_len = chunk_input_len + pad
    numer = padded_len - n_fft
    t_steps = (numer // stride) + 1
    if t_steps <= 0:
        raise ValueError("invalid t_steps")

    # Symbolic batch.
    dim_symbols = ["batch"]
    # dim_exprs: one expr 0 = symbol 0.
    dim_exprs_symbol_indices = [0]
    batch_expr_index = 0

    initializers: List[Initializer] = []
    values: List[ValueRecord] = []
    nodes: List[NodeRecord] = []
    debug_names: List[DebugName] = []

    # Keep rank bookkeeping just for sanity / signature correctness.
    ranks: Dict[int, int] = {}

    def add_public_input(name: str, shape: Tuple[int, ...], batch_symbol: bool) -> int:
        vid = len(values)
        st = _shape_terms_for(shape, batch_expr_index if batch_symbol else None)
        values.append(
            ValueRecord(dtype=DType.f32, rank=len(shape), source=ValueSource.public_input, shape_terms=st, initializer_index=None)
        )
        ranks[vid] = len(shape)
        debug_names.append(DebugName(value=vid, name=name))
        return vid

    def add_initializer_value(name: str, array: np.ndarray) -> int:
        nonlocal initializers
        arr = np.ascontiguousarray(array.astype(np.float32, copy=False))
        data = arr.astype("<f4", copy=False).tobytes(order="C")
        init_idx = len(initializers)
        initializers.append(Initializer(dtype=DType.f32, data=data))

        vid = len(values)
        st = _shape_terms_for(tuple(int(x) for x in arr.shape), None)
        values.append(
            ValueRecord(dtype=DType.f32, rank=arr.ndim, source=ValueSource.initializer, shape_terms=st, initializer_index=init_idx)
        )
        ranks[vid] = arr.ndim
        debug_names.append(DebugName(value=vid, name=name))
        return vid

    def add_produced_value(rank: int) -> int:
        vid = len(values)
        values.append(ValueRecord(dtype=DType.f32, rank=rank, source=ValueSource.produced, shape_terms=_u32(0), initializer_index=None))
        ranks[vid] = rank
        return vid

    def emit(kind: int, inputs: List[int], attr: bytes, out_rank: int) -> int:
        out_vid = add_produced_value(out_rank)
        nodes.append(NodeRecord(kind=kind, output=out_vid, inputs=inputs, attr=attr))
        return out_vid

    # Inputs
    x = add_public_input("x", (1, chunk_input_len), batch_symbol=True)
    h = add_public_input("h", (1, 128), batch_symbol=True)
    c = add_public_input("c", (1, 128), batch_symbol=True)

    # Weights (as initializer-backed values)
    stft_w = add_initializer_value("stft_w", weights["stft_w"])

    conv1_w = add_initializer_value("conv1_w", weights["conv1_w"])
    conv1_b = add_initializer_value("conv1_b", weights["conv1_b"])
    conv2_w = add_initializer_value("conv2_w", weights["conv2_w"])
    conv2_b = add_initializer_value("conv2_b", weights["conv2_b"])
    conv3_w = add_initializer_value("conv3_w", weights["conv3_w"])
    conv3_b = add_initializer_value("conv3_b", weights["conv3_b"])
    conv4_w = add_initializer_value("conv4_w", weights["conv4_w"])
    conv4_b = add_initializer_value("conv4_b", weights["conv4_b"])

    lstm_w_ih = add_initializer_value("lstm_w_ih", weights["lstm_w_ih"])
    lstm_w_hh = add_initializer_value("lstm_w_hh", weights["lstm_w_hh"])
    lstm_b_ih = add_initializer_value("lstm_b_ih", weights["lstm_b_ih"])
    lstm_b_hh = add_initializer_value("lstm_b_hh", weights["lstm_b_hh"])

    final_w = add_initializer_value("final_w", weights["final_w"])
    final_b = add_initializer_value("final_b", weights["final_b"])

    # Forward graph (matches examples/silero_vad.zig)
    x3 = emit(NodeKind.ViewUnsqueeze, [x], _node_attr_unsqueeze(2), out_rank=3)

    stft = emit(
        NodeKind.Conv1D,
        [x3, stft_w],
        _node_attr_conv1d(stride=128, dilation=1, pad_left=0, pad_right=64, pad_mode=pad_mode, groups=1),
        out_rank=3,
    )

    # real/imag slices with dynamic batch lens via shape terms.
    lens3 = [_shape_term_expr(batch_expr_index), _shape_term_constant(t_steps), _shape_term_constant(cutoff)]
    real = emit(NodeKind.ViewSliceND, [stft], _node_attr_slice([0, 0, 0], lens3), out_rank=3)
    imag = emit(NodeKind.ViewSliceND, [stft], _node_attr_slice([0, 0, cutoff], lens3), out_rank=3)

    r2 = emit(NodeKind.ElemwiseBinary, [real, real], _node_attr_binary(ElemwiseBinaryOp.mul), out_rank=3)
    im2 = emit(NodeKind.ElemwiseBinary, [imag, imag], _node_attr_binary(ElemwiseBinaryOp.mul), out_rank=3)
    mag2 = emit(NodeKind.ElemwiseBinary, [r2, im2], _node_attr_binary(ElemwiseBinaryOp.add), out_rank=3)
    mag = emit(NodeKind.Unary, [mag2], _node_attr_unary(UnaryOp.sqrt), out_rank=3)

    conv1_out = emit(
        NodeKind.Conv1D,
        [mag, conv1_w, conv1_b],
        _node_attr_conv1d(stride=1, dilation=1, pad_left=1, pad_right=1, pad_mode=PadMode.zero, groups=1),
        out_rank=3,
    )
    c1 = emit(NodeKind.Unary, [conv1_out], _node_attr_unary(UnaryOp.relu), out_rank=3)

    conv2_out = emit(
        NodeKind.Conv1D,
        [c1, conv2_w, conv2_b],
        _node_attr_conv1d(stride=2, dilation=1, pad_left=1, pad_right=1, pad_mode=PadMode.zero, groups=1),
        out_rank=3,
    )
    c2 = emit(NodeKind.Unary, [conv2_out], _node_attr_unary(UnaryOp.relu), out_rank=3)

    conv3_out = emit(
        NodeKind.Conv1D,
        [c2, conv3_w, conv3_b],
        _node_attr_conv1d(stride=2, dilation=1, pad_left=1, pad_right=1, pad_mode=PadMode.zero, groups=1),
        out_rank=3,
    )
    c3 = emit(NodeKind.Unary, [conv3_out], _node_attr_unary(UnaryOp.relu), out_rank=3)

    conv4_out = emit(
        NodeKind.Conv1D,
        [c3, conv4_w, conv4_b],
        _node_attr_conv1d(stride=1, dilation=1, pad_left=1, pad_right=1, pad_mode=PadMode.zero, groups=1),
        out_rank=3,
    )
    c4 = emit(NodeKind.Unary, [conv4_out], _node_attr_unary(UnaryOp.relu), out_rank=3)

    features = emit(NodeKind.ViewSqueeze, [c4], _node_attr_squeeze(1), out_rank=2)

    packed_state = emit(
        NodeKind.LSTMCell,
        [features, h, c, lstm_w_ih, lstm_w_hh, lstm_b_ih, lstm_b_hh],
        _node_attr_lstm(True),
        out_rank=2,
    )

    # Slice packed state [batch, 2*hidden] into h and c
    lens2_h = [_shape_term_expr(batch_expr_index), _shape_term_constant(128)]
    h_t = emit(NodeKind.ViewSliceND, [packed_state], _node_attr_slice([0, 0], lens2_h), out_rank=2)
    c_t = emit(NodeKind.ViewSliceND, [packed_state], _node_attr_slice([0, 128], lens2_h), out_rank=2)

    h3 = emit(NodeKind.ViewUnsqueeze, [h_t], _node_attr_unsqueeze(1), out_rank=3)
    act = emit(NodeKind.Unary, [h3], _node_attr_unary(UnaryOp.relu), out_rank=3)

    logits3 = emit(
        NodeKind.Conv1D,
        [act, final_w, final_b],
        _node_attr_conv1d(stride=1, dilation=1, pad_left=0, pad_right=0, pad_mode=PadMode.zero, groups=1),
        out_rank=3,
    )

    probs3 = emit(NodeKind.Unary, [logits3], _node_attr_unary(UnaryOp.sigmoid), out_rank=3)
    probs2 = emit(NodeKind.ViewSqueeze, [probs3], _node_attr_squeeze(2), out_rank=2)
    probs1 = emit(NodeKind.Reduce, [probs2], _node_attr_reduce(ReduceOp.mean, axis=1), out_rank=1)
    probs = emit(NodeKind.ViewUnsqueeze, [probs1], _node_attr_unsqueeze(1), out_rank=2)

    # Signatures
    inputs = [NamedValue("x", x), NamedValue("h", h), NamedValue("c", c)]
    outputs = [NamedValue("prob", probs), NamedValue("next_h", h_t), NamedValue("next_c", c_t)]

    # I/O aliases: h -> next_h, c -> next_c (indices within signatures)
    io_aliases = [IoAlias(input_index=1, output_index=1), IoAlias(input_index=2, output_index=2)]

    metadata = [MetadataEntry(key="arch", value="tiny-silero-vad")]

    return Package(
        initializers=initializers,
        values=values,
        nodes=nodes,
        inputs=inputs,
        outputs=outputs,
        dim_symbols=dim_symbols,
        dim_exprs_symbol_indices=dim_exprs_symbol_indices,
        metadata=metadata,
        debug_names=debug_names,
        io_aliases=io_aliases,
    )


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description="Convert silero VAD safetensors to Aion .aion")
    ap.add_argument("in_safetensors", type=str)
    ap.add_argument("out_aion", type=str)
    ap.add_argument("--num-samples", type=int, default=512)
    ap.add_argument("--context", type=int, default=64)
    ap.add_argument("--pad-mode", type=str, default="reflect", choices=("reflect", "zero"))
    ap.add_argument("--print-keys", action="store_true", help="Print all safetensors keys and exit")
    args = ap.parse_args(argv)

    state = _load_safetensors_numpy(args.in_safetensors)
    print(f"loaded {args.in_safetensors} with {len(state)} tensors")

    if args.print_keys:
        for k in sorted(state.keys()):
            v = state[k]
            print(f"  {k}: dtype={v.dtype} shape={v.shape}")
        return 0

    w = extract_weights(state)
    pad_mode = PadMode.reflect if args.pad_mode == "reflect" else PadMode.zero

    pkg = build_tiny_silero_aion_package(w, num_samples=args.num_samples, context_size=args.context, pad_mode=pad_mode)
    write_aion_v4(args.out_aion, pkg)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
