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
"""INTERNAL graph builder over `aion._writer.format`. No stability guarantees.

One ergonomic layer for constructing `.aion` packages: values, initializers
(f32/i32/q8_0), node emitters mirroring `src/aion/graph/infer.zig` dtype rules,
control-flow regions/loops, io-aliases, and input roles. Unifies the builder
classes that previously lived copy-pasted inside each converter script.

Shapes accept `int` (constant dims) and `str` (dim symbols registered via the
constructor's `dim_symbols`). Method names come in two flavors kept as exact
aliases — descriptive (`add_f32_initializer`, `elemwise`, ...) and short
(`f32`, `ew`, ...) — so existing converters keep working unchanged.

This module's method surface doubles as a design draft for the future
model-building API (the core-lib Zig `Builder` exposed through the C ABI),
which will eventually replace it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import format as aw

Dim = Union[int, str]
Shape = Tuple[Dim, ...]


class Builder:
    """Holds an `aw.Package` and exposes semantic helpers for values/nodes/initializers.

    Bookkeeping invariants: every value carries a rank and concrete dtype (the
    on-disk validator requires rank > 0; signatures/io-alias compatibility rely
    on dtype being correct), tracked in `_rank`/`_dtype` as values are created.
    """

    def __init__(self, *, dim_symbols: Sequence[str] = (), ln_eps: float = 1.0e-5,
                 rms_eps: float = 1.0e-6) -> None:
        self.pkg = aw.Package()
        self.ln_eps = ln_eps
        self.rms_eps = rms_eps
        self._sym_idx: Dict[str, int] = {}
        for sym in dim_symbols:
            self._sym_idx[sym] = len(self.pkg.dim_symbols)
            self.pkg.dim_symbols.append(sym)
            self.pkg.dim_exprs_symbol_indices.append(len(self.pkg.dim_exprs_symbol_indices))
        self._rank: Dict[int, int] = {}
        self._dtype: Dict[int, int] = {}
        # Stack of in-progress control-flow region node lists. While non-empty,
        # `_emit` appends nodes to the top region instead of the main graph.
        self._region_stack: List[List[aw.NodeRecord]] = []
        # Shared constant caches keyed by shape / (dim, value).
        self._zero_beta_by_shape: Dict[Tuple[int, ...], int] = {}
        self._scale_vec_by_key: Dict[Tuple[int, float], int] = {}

    # ---- shape terms ----

    def _term(self, dim: Dim) -> bytes:
        if isinstance(dim, str):
            return aw.shape_term_expr(self._sym_idx[dim])
        return aw.shape_term_constant(int(dim))

    def _terms(self, dims: Sequence[Dim]) -> List[bytes]:
        return [self._term(d) for d in dims]

    def _shape_terms(self, shape: Sequence[Dim]) -> bytes:
        out = bytearray(aw.u32(len(shape)))
        for d in shape:
            out += self._term(d)
        return bytes(out)

    # ---- io / signatures ----

    def add_input(self, name: str, dtype: int, shape: Sequence[Dim]) -> int:
        vid = len(self.pkg.values)
        self.pkg.values.append(aw.ValueRecord(dtype=dtype, rank=len(shape),
                                              source=aw.ValueSource.public_input,
                                              shape_terms=self._shape_terms(shape)))
        self.pkg.inputs.append(aw.NamedValue(name=name, value=vid))
        self.pkg.debug_names.append(aw.DebugName(value=vid, name=name))
        self._rank[vid] = len(shape)
        self._dtype[vid] = dtype
        return vid

    def add_output(self, name: str, value_id: int) -> None:
        self.pkg.outputs.append(aw.NamedValue(name=name, value=value_id))

    def add_io_alias(self, input_index: int, output_index: int) -> None:
        self.pkg.io_aliases.append(aw.IoAlias(input_index, output_index))

    def input_index(self, name: str) -> int:
        return next(i for i, nv in enumerate(self.pkg.inputs) if nv.name == name)

    def output_index(self, name: str) -> int:
        return next(i for i, nv in enumerate(self.pkg.outputs) if nv.name == name)

    def add_io_alias_by_name(self, input_name: str, output_name: str) -> None:
        self.add_io_alias(self.input_index(input_name), self.output_index(output_name))

    def add_input_role(self, input_name: str, kind: int, *, axis: int = aw.ROLE_NO_AXIS,
                       flags: int = aw.ROLE_FLAG_ZERO_INIT,
                       capacity_symbol: int = aw.INVALID_INDEX_U32) -> None:
        self.pkg.input_roles.append(aw.InputRole(input_index=self.input_index(input_name),
                                                 kind=kind, axis=axis, flags=flags,
                                                 capacity_symbol=capacity_symbol))

    # ---- initializers ----

    def _add_init(self, name: str, init: aw.Initializer, dtype: int,
                  shape: Tuple[int, ...]) -> int:
        idx = len(self.pkg.initializers)
        self.pkg.initializers.append(init)
        vid = len(self.pkg.values)
        st = bytearray(aw.u32(len(shape)))
        for d in shape:
            st += aw.shape_term_constant(int(d))
        self.pkg.values.append(aw.ValueRecord(dtype=dtype, rank=len(shape),
                                              source=aw.ValueSource.initializer,
                                              shape_terms=bytes(st), initializer_index=idx))
        self.pkg.debug_names.append(aw.DebugName(value=vid, name=name))
        self._rank[vid] = len(shape)
        self._dtype[vid] = dtype
        return vid

    def add_f32_initializer(self, name: str, arr: np.ndarray) -> int:
        arr = np.ascontiguousarray(np.asarray(arr).astype(np.float32))
        return self._add_init(name, aw.Initializer.plain(aw.DType.f32, arr.tobytes(order="C")),
                              aw.DType.f32, tuple(int(x) for x in arr.shape))

    def add_i32_initializer(self, name: str, arr: np.ndarray) -> int:
        arr = np.ascontiguousarray(np.asarray(arr).astype(np.int32))
        return self._add_init(name, aw.Initializer.plain(aw.DType.i32, arr.tobytes(order="C")),
                              aw.DType.i32, tuple(int(x) for x in arr.shape))

    def add_q8_0_matmul_b(self, name: str, w_torch: np.ndarray) -> int:
        """PyTorch linear `[out, in]` -> Aion matmul-B `[1, in, out]` packed q8_0 (quant_axis=1).

        Aion `MatMul` expects A and B to have the same rank; residual-stream A is
        rank-3 `[B, S, K]`, so dense weights are stored `[1, K, N]` and broadcast.
        """
        w_t = np.ascontiguousarray(w_torch.T)  # [K, N]
        if w_t.shape[0] % aw.Q8_0_BLOCK_ELEMS != 0:
            raise ValueError(f"{name}: K={w_t.shape[0]} not a multiple of {aw.Q8_0_BLOCK_ELEMS}")
        packed = aw.pack_q8_0(w_t.reshape(1, w_t.shape[0], w_t.shape[1]), axis=1)
        init = aw.Initializer.quantized_q8_0(logical_dtype=aw.DType.f16, quant_axis=1, data=packed)
        return self._add_init(name, init, aw.DType.q8_0, (1, int(w_t.shape[0]), int(w_t.shape[1])))

    def add_q8_0_embedding(self, name: str, table: np.ndarray) -> int:
        if table.shape[1] % aw.Q8_0_BLOCK_ELEMS != 0:
            raise ValueError(f"{name}: embed dim {table.shape[1]} not a multiple of {aw.Q8_0_BLOCK_ELEMS}")
        packed = aw.pack_q8_0(np.ascontiguousarray(table.astype(np.float32)), axis=1)
        init = aw.Initializer.quantized_q8_0(logical_dtype=aw.DType.f16, quant_axis=1, data=packed)
        return self._add_init(name, init, aw.DType.q8_0, (int(table.shape[0]), int(table.shape[1])))

    def zero_beta(self, shape: Tuple[int, ...]) -> int:
        """Shared all-zeros f32 constant for norm ops without a beta, cached per shape."""
        key = tuple(int(x) for x in shape)
        if key not in self._zero_beta_by_shape:
            self._zero_beta_by_shape[key] = self.add_f32_initializer(
                f"_zero_beta_{'x'.join(str(x) for x in key)}", np.zeros(key, dtype=np.float32))
        return self._zero_beta_by_shape[key]

    def scale_broadcast_vec(self, dim: int, value: float) -> int:
        """A constant `[dim]` f32 vector of `value` used for last-dim scalar scaling."""
        key = (dim, float(value))
        if key not in self._scale_vec_by_key:
            self._scale_vec_by_key[key] = self.add_f32_initializer(
                f"_scale_{dim}_{value:g}", np.full((dim,), value, dtype=np.float32))
        return self._scale_vec_by_key[key]

    # ---- node emission core ----

    def _dtype_of(self, value_id: int) -> int:
        return self._dtype[value_id]

    def _produced(self, dtype: int, rank: int) -> int:
        if rank <= 0:
            raise ValueError(f"produced value must have rank >= 1 (got {rank})")
        vid = len(self.pkg.values)
        self.pkg.values.append(aw.ValueRecord(dtype=dtype, rank=rank,
                                              source=aw.ValueSource.produced, shape_terms=aw.u32(0)))
        self._rank[vid] = rank
        self._dtype[vid] = dtype
        return vid

    def _emit(self, kind: int, inputs: List[int], attr: bytes, *, dtype: int, rank: int) -> int:
        out = self._produced(dtype, rank)
        rec = aw.NodeRecord(kind=kind, output=out, inputs=list(inputs), attr=attr)
        if self._region_stack:
            self._region_stack[-1].append(rec)
        else:
            self.pkg.nodes.append(rec)
        return out

    # ---- control-flow region API ----

    def begin_region(self) -> None:
        self._region_stack.append([])

    def end_region(self, outputs: List[int]) -> int:
        nodes = self._region_stack.pop()
        rid = len(self.pkg.regions)
        self.pkg.regions.append(aw.RegionRecord(nodes=nodes, outputs=list(outputs)))
        return rid

    def loop(self, carried: List[int], body_region: int, trip: int, *,
             cond_carry: Optional[int] = None, check_before: bool = True) -> List[int]:
        """Multi-carry loop. `carried[i]` pairs with body-region output i; returns
        the N final carried value ids. `cond_carry`, if set, names the carry index
        (an i32 [1] value) used as the continue predicate for early exit."""
        outs = [self._produced(self._dtype[c], self._rank[c]) for c in carried]
        attr = aw.attr_loop(body_region, trip, cond_carry=cond_carry,
                            check_before=check_before, extra_outputs=outs[1:])
        rec = aw.NodeRecord(kind=aw.NodeKind.Loop, output=outs[0], inputs=list(carried), attr=attr)
        if self._region_stack:
            self._region_stack[-1].append(rec)
        else:
            self.pkg.nodes.append(rec)
        return outs

    # ---- matmuls ----

    def matmul(self, a: int, b: int) -> int:
        # A[..., K] @ B[..., K, N] -> [..., N]; rank preserved from A.
        # Mirrors `src/aion/graph/infer.zig` dtype routing for MatMul.
        a_dt, b_dt = self._dtype_of(a), self._dtype_of(b)
        if b_dt in (aw.DType.q4_0, aw.DType.q8_0):
            if a_dt != aw.DType.f32:
                raise ValueError("MatMul: quantized B requires f32 A")
            out_dt = aw.DType.f32
        elif b_dt == aw.DType.f32:
            if a_dt != aw.DType.f32:
                raise ValueError("MatMul: f32 B requires f32 A")
            out_dt = aw.DType.f32
        elif b_dt == aw.DType.f16:
            if a_dt != aw.DType.f16:
                raise ValueError("MatMul: f16 B requires f16 A")
            out_dt = aw.DType.f16
        else:
            raise ValueError(f"MatMul: unsupported B dtype: {b_dt}")
        return self._emit(aw.NodeKind.MatMul, [a, b], aw.attr_matmul(), dtype=out_dt, rank=self._rank[a])

    def matmul_nt(self, a: int, b: int) -> int:
        # Mirrors `infer.zig:MatMulNT` (tied-logits path).
        if self._dtype_of(a) != aw.DType.f32:
            raise ValueError("MatMulNT: A must be f32")
        if self._dtype_of(b) != aw.DType.q8_0:
            raise ValueError("MatMulNT: B must be q8_0")
        return self._emit(aw.NodeKind.MatMulNT, [a, b], aw.attr_matmul_nt(),
                          dtype=aw.DType.f32, rank=self._rank[a])

    # ---- elementwise / broadcast ----

    def elemwise(self, op: int, a: int, b: int) -> int:
        """Generic elementwise binary (arithmetic); output dtype follows the inputs."""
        a_dt, b_dt = self._dtype_of(a), self._dtype_of(b)
        if a_dt != b_dt:
            raise ValueError("ElemwiseBinary: dtype mismatch")
        if a_dt in (aw.DType.q4_0, aw.DType.q8_0):
            raise ValueError("ElemwiseBinary: quantized inputs are not supported")
        return self._emit(aw.NodeKind.ElemwiseBinary, [a, b], aw.attr_binary(op),
                          dtype=a_dt, rank=self._rank[a])

    def compare(self, op: int, a: int, b: int) -> int:
        # i32 comparison -> i32 {0,1}
        return self._emit(aw.NodeKind.ElemwiseBinary, [a, b], aw.attr_binary(op),
                          dtype=aw.DType.i32, rank=self._rank[a])

    def add(self, a: int, b: int) -> int:
        return self.elemwise(aw.ElemwiseBinaryOp.add, a, b)

    def mul(self, a: int, b: int) -> int:
        return self.elemwise(aw.ElemwiseBinaryOp.mul, a, b)

    def broadcast_last_binary(self, op: int, a: int, vec: int) -> int:
        a_dt = self._dtype_of(a)
        if a_dt != self._dtype_of(vec):
            raise ValueError("BroadcastLastDimBinary: dtype mismatch")
        if a_dt in (aw.DType.q4_0, aw.DType.q8_0):
            raise ValueError("BroadcastLastDimBinary: quantized inputs are not supported")
        return self._emit(aw.NodeKind.BroadcastLastDimBinary, [a, vec], aw.attr_binary(op),
                          dtype=a_dt, rank=self._rank[a])

    def bcast_add(self, a: int, vec: int) -> int:
        return self.broadcast_last_binary(aw.ElemwiseBinaryOp.add, a, vec)

    def bcast_mul(self, a: int, vec: int) -> int:
        return self.broadcast_last_binary(aw.ElemwiseBinaryOp.mul, a, vec)

    def scalar_last_dim_mul(self, x: int, dim: int, value: float) -> int:
        return self.broadcast_last_binary(aw.ElemwiseBinaryOp.mul, x, self.scale_broadcast_vec(dim, value))

    def scalar_last_dim_div(self, x: int, dim: int, value: float) -> int:
        return self.broadcast_last_binary(aw.ElemwiseBinaryOp.div, x, self.scale_broadcast_vec(dim, value))

    def unary(self, op: int, a: int) -> int:
        a_dt = self._dtype_of(a)
        if a_dt in (aw.DType.q4_0, aw.DType.q8_0):
            raise ValueError("Unary: quantized inputs are not supported")
        return self._emit(aw.NodeKind.Unary, [a], aw.attr_unary(op), dtype=a_dt, rank=self._rank[a])

    # ---- norms ----

    def layernorm(self, x: int, gamma: int, beta: int, normalized_shape: Tuple[int, ...],
                  eps: Optional[float] = None) -> int:
        terms = [aw.shape_term_constant(int(d)) for d in normalized_shape]
        return self._emit(aw.NodeKind.LayerNorm, [x, gamma, beta],
                          aw.attr_layernorm(self.ln_eps if eps is None else eps, terms),
                          dtype=self._dtype[x], rank=self._rank[x])

    def rmsnorm(self, x: int, gamma: int, beta: int, normalized_shape: Tuple[int, ...],
                eps: Optional[float] = None) -> int:
        x_dt = self._dtype_of(x)
        if self._dtype_of(gamma) != x_dt or self._dtype_of(beta) != x_dt:
            raise ValueError("RMSNorm: dtype mismatch")
        if x_dt in (aw.DType.q4_0, aw.DType.q8_0):
            raise ValueError("RMSNorm: quantized inputs are not supported")
        terms = [aw.shape_term_constant(int(d)) for d in normalized_shape]
        return self._emit(aw.NodeKind.RMSNorm, [x, gamma, beta],
                          aw.attr_rmsnorm(self.rms_eps if eps is None else eps, terms),
                          dtype=x_dt, rank=self._rank[x])

    # ---- convolutions / signal ----

    def conv1d(self, x: int, w: int, bias: Optional[int], *, stride: int, dilation: int,
               pad_left: int, pad_right: int, groups: int) -> int:
        attr = aw.attr_conv1d(stride, dilation, pad_left, pad_right, aw.PadMode.zero, groups)
        ins = [x, w] if bias is None else [x, w, bias]
        return self._emit(aw.NodeKind.Conv1D, ins, attr, dtype=aw.DType.f32, rank=3)

    def conv2d(self, x: int, w: int, bias: Optional[int], *, stride: int,
               pad_top: int, pad_bottom: int, pad_left: int, pad_right: int, groups: int) -> int:
        attr = aw.attr_conv2d(stride, stride, 1, 1, pad_top, pad_bottom, pad_left, pad_right,
                              aw.PadMode.zero, groups)
        ins = [x, w] if bias is None else [x, w, bias]
        return self._emit(aw.NodeKind.Conv2D, ins, attr, dtype=aw.DType.f32, rank=4)

    def stft(self, signal: int, window: int, n_fft: int, hop: int, center: bool) -> int:
        # signal [1, samples], window [n_fft] -> [1, frames, n_fft+2] (split re|im).
        return self._emit(aw.NodeKind.STFT, [signal, window], aw.attr_stft(n_fft, hop, center),
                          dtype=aw.DType.f32, rank=3)

    # ---- views ----

    def reshape(self, x: int, new_shape: Sequence[Dim]) -> int:
        return self._emit(aw.NodeKind.ViewReshape, [x], aw.attr_view_reshape(self._terms(new_shape)),
                          dtype=self._dtype[x], rank=len(new_shape))

    def slice_nd(self, x: int, starts: Sequence[int], lens: Sequence[Dim]) -> int:
        return self._emit(aw.NodeKind.ViewSliceND, [x],
                          aw.attr_view_slice_nd(list(starts), self._terms(lens)),
                          dtype=self._dtype[x], rank=len(lens))

    def concat(self, inputs: List[int], axis: int, out_rank: int) -> int:
        return self._emit(aw.NodeKind.Concat, list(inputs), aw.attr_concat(axis),
                          dtype=self._dtype[inputs[0]], rank=out_rank)

    # ---- attention / recurrent / indexed ----

    def relpos_mha(self, q: int, k: int, v: int, pos_emb: int, bu: int, bv: int,
                   mask: Optional[int], scale: float, heads: int) -> int:
        ins = [q, k, v, pos_emb, bu, bv] + ([mask] if mask is not None else [])
        attr = aw.attr_relpos_mha(scale, heads, mask is not None)
        return self._emit(aw.NodeKind.RelPosMHA, ins, attr, dtype=aw.DType.f32, rank=4)

    def mha_cached(self, q: int, k: int, v: int, positions: int, end_index: int,
                   scale: float, sliding_window: int, softcap: float) -> int:
        return self._emit(
            aw.NodeKind.MultiHeadAttentionCached, [q, k, v, positions, end_index],
            aw.attr_mha_cached(scale=scale, causal=True, sliding_window=sliding_window,
                               attn_logits_soft_cap=softcap),
            dtype=aw.DType.f32, rank=self._rank[q])

    def rope1d(self, x: int, positions: int, base: float, *, scale_factor: float = 1.0,
               rope_proportion: float = 1.0) -> int:
        return self._emit(
            aw.NodeKind.RoPE1D, [x, positions],
            aw.attr_rope1d(base_frequency=base, scale_factor=scale_factor,
                           rope_proportion=rope_proportion),
            dtype=self._dtype[x], rank=self._rank[x])

    def sequence_append(self, cache: int, new_kv: int, end_index: int) -> int:
        cache_dt = self._dtype_of(cache)
        if self._dtype_of(new_kv) != cache_dt:
            raise ValueError("SequenceAppend: cache/new_kv dtype mismatch")
        if cache_dt not in (aw.DType.f16, aw.DType.f32):
            raise ValueError("SequenceAppend: unsupported cache dtype")
        if self._dtype_of(end_index) != aw.DType.i32:
            raise ValueError("SequenceAppend: end_index must be i32")
        return self._emit(aw.NodeKind.SequenceAppend, [cache, new_kv, end_index], b"",
                          dtype=cache_dt, rank=self._rank[cache])

    def gather_rows(self, table: int, indices: int) -> int:
        table_dt = self._dtype_of(table)
        if self._dtype_of(indices) != aw.DType.i32:
            raise ValueError("GatherRows: indices must be i32")
        if table_dt in (aw.DType.f16, aw.DType.f32):
            out_dt = table_dt
        elif table_dt == aw.DType.q8_0:
            out_dt = aw.DType.f32  # mirrors infer.zig: q8_0 table materializes f32
        else:
            raise ValueError("GatherRows: unsupported table dtype")
        return self._emit(aw.NodeKind.GatherRows, [table, indices], b"", dtype=out_dt, rank=3)

    def lstm_cell(self, x: int, h: int, c: int, w_ih: int, w_hh: int, b_ih: int, b_hh: int) -> int:
        # out [batch, 2*hidden] = (h_t | c_t)
        return self._emit(aw.NodeKind.LSTMCell, [x, h, c, w_ih, w_hh, b_ih, b_hh],
                          aw.attr_lstm(True), dtype=aw.DType.f32, rank=2)

    def argmax(self, x: int, axis: int, out_rank: int) -> int:
        return self._emit(aw.NodeKind.ArgMax, [x], aw.attr_argmax(axis),
                          dtype=aw.DType.i32, rank=out_rank)

    def scatter_row(self, buf: int, idx: int, src: int) -> int:
        return self._emit(aw.NodeKind.ScatterRow, [buf, idx, src], b"",
                          dtype=self._dtype[buf], rank=self._rank[buf])

    def cast(self, x: int, to_dtype: int) -> int:
        x_dt = self._dtype_of(x)
        # Runtime-supported casts (exec/cast.zig): identity, f16<->f32, f32<->i32.
        ok = (x_dt == to_dtype
              or {x_dt, to_dtype} == {aw.DType.f16, aw.DType.f32}
              or {x_dt, to_dtype} == {aw.DType.f32, aw.DType.i32})
        if not ok:
            raise ValueError(f"Cast: unsupported cast {x_dt} -> {to_dtype}")
        return self._emit(aw.NodeKind.Cast, [x], aw.attr_cast(to_dtype),
                          dtype=to_dtype, rank=self._rank[x])

    # ---- output ----

    def write(self, path: str) -> None:
        aw.write_aion_v4(path, self.pkg)

    # ---- short aliases (converter-facing; same functions, terser names) ----

    f32 = add_f32_initializer
    i32_const = add_i32_initializer
    q8_matmul_b = add_q8_0_matmul_b
    q8_embedding = add_q8_0_embedding
    ew = elemwise
