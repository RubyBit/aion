#!/usr/bin/env python3
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
"""Convert Gemma 4 E2B text-only safetensors -> AION v4 package (full forward pass).

REFERENCE: https://github.com/rwightman/gemma4_pytorch_claude
SPEC:      docs/GEMMA4_E2B_TEXT_PLAN.md

Quantization policy:
- Dense matmul weights  : q8_0, quant_axis=1 (matmul-B layout `[1, K, N]`; blocks along K).
- Embedding tables      : q8_0, quant_axis=1 (per-row blocks; tied-logits-friendly).
- RMSNorm γ vectors     : f32 (required by RMSNorm op's dtype contract on f32 residuals).
- RMSNorm β (zeros)     : f32 shared per-shape (Gemma has no β; satisfies op signature).
- Scaling / softcap vec : f32 broadcast vectors (sized to match the last axis they multiply).
- KV caches             : f16 public inputs; new_k/new_v are cast f32->f16 before append.

KV sharing:
- Layers 0..14 own cache tensors. Layers 15..34 reuse the source-layer cache output
  (the value id produced by the source layer's KVCacheAppend) — no K/V materialized
  for those layers, and their k_proj/v_proj/k_norm weights are skipped.

Forward pass (abridged, full detail inline):
  embeddings + PLI encoder
  -> 35x per-block:
       pre-attn RMSNorm
    q_proj, [optional] k_proj, v_proj  (regular MatMul; B stored as [1, K, N] q8_0)
       q_norm / k_norm (RMSNorm over head_dim)
       RoPE1D on q, [optional] k
       [source layers] cast f32->f16 new_k/new_v, KVCacheAppend
       MultiHeadAttentionCached (selects source cache for KV-sharing layers)
      o_proj, post-attn RMSNorm, residual
      FFW: pre-ffn RMSNorm, gate/up MatMul, gelu_tanh, multiply, down MatMul, post-ffn RMSNorm, residual
       PLI mapping: gate MatMul, gelu_tanh, multiply with per-layer pli slice, proj MatMul, RMSNorm, residual
      skip_scale: scalar multiplier at end of block
  tail:
       final RMSNorm
       tied logits via MatMulNT on the token embedding (no duplication)
       softcap: tanh(x/30)*30 via BroadcastLastDimBinary divide then tanh then mul

Usage:
  uv run --project bindings/python scripts/convert_gemma4_e2b_to_aion.py \\
      models/gemma/model.safetensors models/gemma/gemma4_e2b_q8.aion
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _aion_writer as aw  # type: ignore  # noqa: E402


# ------------------------------ Model constants ------------------------------

NUM_LAYERS: int = 35
EMBED_DIM: int = 1536
VOCAB_SIZE: int = 262144
NUM_HEADS: int = 8
NUM_KV_HEADS: int = 1
LOCAL_HEAD_DIM: int = 256
GLOBAL_HEAD_DIM: int = 512
LOCAL_SLIDING_WINDOW: int = 512
FINAL_LOGIT_SOFTCAP: float = 30.0
PLI_DIM: int = 256
PLI_TOTAL: int = NUM_LAYERS * PLI_DIM  # 8960
FFN_DIM: int = 6144
RMS_EPS: float = 1e-6
# Rotary base frequencies. Gemma's upstream spec uses different bases for local vs global
# attention; values here mirror the PyTorch reference.
ROPE_LOCAL_BASE: float = 10_000.0
ROPE_GLOBAL_BASE: float = 1_000_000.0
ROPE_LOCAL_PROPORTION: float = 1.0
ROPE_GLOBAL_PROPORTION: float = 0.25


def is_global_layer(layer: int) -> bool:
    return (layer % 5) == 4


LOCAL_SOURCE_LAYER: int = 13
GLOBAL_SOURCE_LAYER: int = 14
SOURCE_LAYERS: Tuple[int, ...] = tuple(range(15))


def kv_source_of(layer: int) -> int:
    if layer in SOURCE_LAYERS:
        return layer
    return GLOBAL_SOURCE_LAYER if is_global_layer(layer) else LOCAL_SOURCE_LAYER


# ------------------------------- CLI + loading -------------------------------


def _parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert Gemma 4 E2B text-only to AION v4")
    ap.add_argument("in_safetensors", type=str)
    ap.add_argument("out_aion", type=str)
    ap.add_argument(
        "--debug-block-layer",
        type=int,
        default=None,
        help=(
            "Emit a minimal package for a single transformer block (source layers 0..14 only). "
            "Inputs: x (f32), positions (i32), cache_write_index (i32), cache_visible_end (i32), "
            "k_cache (f32/f16), v_cache (f32/f16), per_layer_input (f32). "
            "Outputs: x_out, next_k_cache, next_v_cache."
        ),
    )
    ap.add_argument(
        "--debug-block-weights",
        type=str,
        choices=["q8", "f32"],
        default="q8",
        help="Weight storage for --debug-block-layer (default: q8).",
    )
    ap.add_argument(
        "--debug-block-cache-dtype",
        type=str,
        choices=["f16", "f32"],
        default="f32",
        help="KV cache dtype for --debug-block-layer (default: f32).",
    )
    ap.add_argument(
        "--debug-block-taps",
        action="store_true",
        help="When set, the debug single-block package will also output intermediate tensors for parity debugging.",
    )
    ap.add_argument(
        "--allow-multimodal",
        action="store_true",
        help="Ignore audio/vision tensors in a multimodal checkpoint (they are not included).",
    )
    return ap.parse_args(argv)


def _add_f32_matmul_b(b: "_PackageBuilder", name: str, w_torch_layout: np.ndarray) -> int:
    """Store a PyTorch linear weight `[out, in]` as an f32 MatMul-B tensor `[1, in, out]`."""
    w_t: np.ndarray = np.ascontiguousarray(w_torch_layout.astype(np.float32, copy=False).T)
    w_t3: np.ndarray = w_t.reshape((1, int(w_t.shape[0]), int(w_t.shape[1])))
    return b.add_f32_initializer(name, w_t3)


def _emit_one_layer_weights(
    loader: _WeightLoader,
    b: "_PackageBuilder",
    layer_idx: int,
    *,
    weight_mode: str,
) -> _LayerWeights:
    """Load and emit initializers for a single transformer layer (for debug packages)."""
    if layer_idx < 0 or layer_idx >= NUM_LAYERS:
        raise ValueError(f"layer_idx out of range: {layer_idx}")
    if weight_mode not in {"q8", "f32"}:
        raise ValueError(f"unknown weight_mode: {weight_mode}")

    pfx = f"model.language_model.layers.{layer_idx}"

    def name(s: str) -> str:
        return f"layers.{layer_idx}.{s}"

    def f32v(nm: str, arr: np.ndarray) -> int:
        return b.add_f32_initializer(nm, arr)

    def dense(nm: str, w: np.ndarray) -> int:
        if weight_mode == "q8":
            return b.add_q8_0_matmul_b(nm, w)
        return _add_f32_matmul_b(b, nm, w)

    skip_scale_arr: np.ndarray = loader.get_f32(f"{pfx}.layer_scalar")
    if skip_scale_arr.size != 1:
        raise ValueError(f"unexpected layer_scalar shape for layer {layer_idx}: {skip_scale_arr.shape}")
    skip_scale: float = float(skip_scale_arr.reshape(()))

    lw = _LayerWeights(
        input_ln=f32v(name("input_layernorm.weight"), loader.get_f32(f"{pfx}.input_layernorm.weight")),
        post_attn_ln=f32v(name("post_attention_layernorm.weight"), loader.get_f32(f"{pfx}.post_attention_layernorm.weight")),
        pre_ffn_ln=f32v(name("pre_feedforward_layernorm.weight"), loader.get_f32(f"{pfx}.pre_feedforward_layernorm.weight")),
        post_ffn_ln=f32v(name("post_feedforward_layernorm.weight"), loader.get_f32(f"{pfx}.post_feedforward_layernorm.weight")),
        post_pli_ln=f32v(
            name("post_per_layer_input_norm.weight"),
            loader.get_f32(f"{pfx}.post_per_layer_input_norm.weight"),
        ),
        skip_scale=skip_scale,

        q_proj=dense(name("self_attn.q_proj.weight"), loader.get_f32(f"{pfx}.self_attn.q_proj.weight")),
        o_proj=dense(name("self_attn.o_proj.weight"), loader.get_f32(f"{pfx}.self_attn.o_proj.weight")),
        q_norm=f32v(name("self_attn.q_norm.weight"), loader.get_f32(f"{pfx}.self_attn.q_norm.weight")),

        k_proj=dense(name("self_attn.k_proj.weight"), loader.get_f32(f"{pfx}.self_attn.k_proj.weight")),
        v_proj=dense(name("self_attn.v_proj.weight"), loader.get_f32(f"{pfx}.self_attn.v_proj.weight")),
        k_norm=f32v(name("self_attn.k_norm.weight"), loader.get_f32(f"{pfx}.self_attn.k_norm.weight")),

        gate_proj=dense(name("mlp.gate_proj.weight"), loader.get_f32(f"{pfx}.mlp.gate_proj.weight")),
        up_proj=dense(name("mlp.up_proj.weight"), loader.get_f32(f"{pfx}.mlp.up_proj.weight")),
        down_proj=dense(name("mlp.down_proj.weight"), loader.get_f32(f"{pfx}.mlp.down_proj.weight")),

        pli_gate=dense(name("per_layer_input_gate.weight"), loader.get_f32(f"{pfx}.per_layer_input_gate.weight")),
        pli_proj=dense(name("per_layer_projection.weight"), loader.get_f32(f"{pfx}.per_layer_projection.weight")),
    )
    return lw


def _emit_forward_one_block(
    b: "_PackageBuilder",
    *,
    layer_idx: int,
    lw: _LayerWeights,
    cache_dtype: int,
) -> Tuple[int, int, int, Dict[str, int]]:
    """Emit a minimal forward graph for a single source-layer transformer block."""
    if layer_idx not in SOURCE_LAYERS:
        raise ValueError(
            "debug single-block mode currently supports only source layers 0..14 "
            f"(got layer={layer_idx})"
        )

    is_glob: bool = is_global_layer(layer_idx)
    head_dim: int = GLOBAL_HEAD_DIM if is_glob else LOCAL_HEAD_DIM
    sliding: int = 0 if is_glob else LOCAL_SLIDING_WINDOW
    rope_base: float = ROPE_GLOBAL_BASE if is_glob else ROPE_LOCAL_BASE
    rope_prop: float = ROPE_GLOBAL_PROPORTION if is_glob else ROPE_LOCAL_PROPORTION
    t_dim: Union[int, str] = "G" if is_glob else LOCAL_SLIDING_WINDOW

    # Public runtime inputs.
    x = b.add_input("x", aw.DType.f32, ("batch", "seq", EMBED_DIM))
    positions = b.add_input("positions", aw.DType.i32, ("batch", "seq"))
    cache_write_index = b.add_input("cache_write_index", aw.DType.i32, ("batch",))
    cache_visible_end = b.add_input("cache_visible_end", aw.DType.i32, ("batch",))
    per_layer_input = b.add_input("per_layer_input", aw.DType.f32, ("batch", "seq", PLI_DIM))

    k_cache = b.add_input("k_cache", cache_dtype, ("batch", NUM_KV_HEADS, t_dim, head_dim))
    v_cache = b.add_input("v_cache", cache_dtype, ("batch", NUM_KV_HEADS, t_dim, head_dim))

    taps: Dict[str, int] = {}

    # 1) Pre-attn norm on x.
    x_norm = b.rmsnorm(x, lw.input_ln, b.zero_beta((EMBED_DIM,)), (EMBED_DIM,))
    taps["x_norm"] = x_norm

    # 2) Q projection, reshape to [B, S, H, D_k], norm, rope.
    q_flat = b.matmul(x_norm, lw.q_proj)  # [B, S, H*D_k]
    taps["q_flat"] = q_flat
    q = b.reshape(q_flat, ("batch", "seq", NUM_HEADS, head_dim))
    taps["q_reshape"] = q
    q = b.rmsnorm(q, lw.q_norm, b.zero_beta((head_dim,)), (head_dim,))
    taps["q_norm"] = q
    q = b.rope1d(q, positions, rope_base, rope_proportion=rope_prop)
    taps["q_rope"] = q

    # 3) K/V projection, norm, rope, append.
    assert lw.k_proj is not None and lw.v_proj is not None and lw.k_norm is not None
    k_flat = b.matmul(x_norm, lw.k_proj)
    taps["k_flat"] = k_flat
    k4 = b.reshape(k_flat, ("batch", "seq", NUM_KV_HEADS, head_dim))
    taps["k_reshape"] = k4
    k4 = b.rmsnorm(k4, lw.k_norm, b.zero_beta((head_dim,)), (head_dim,))
    taps["k_norm"] = k4
    k4 = b.rope1d(k4, positions, rope_base, rope_proportion=rope_prop)
    taps["k_rope"] = k4
    k4_t = b.reshape(k4, ("batch", NUM_KV_HEADS, "seq", head_dim))
    if cache_dtype == aw.DType.f16:
        k4_t = b.cast(k4_t, aw.DType.f16)
    k_next = b.kv_cache_append(k_cache, k4_t, cache_write_index)

    v_flat = b.matmul(x_norm, lw.v_proj)
    taps["v_flat"] = v_flat
    v4 = b.reshape(v_flat, ("batch", "seq", NUM_KV_HEADS, head_dim))
    taps["v_reshape"] = v4
    v4 = b.rmsnorm(v4, b.scale_broadcast_vec(head_dim, 1.0), b.zero_beta((head_dim,)), (head_dim,))
    taps["v_norm"] = v4
    v4_t = b.reshape(v4, ("batch", NUM_KV_HEADS, "seq", head_dim))
    if cache_dtype == aw.DType.f16:
        v4_t = b.cast(v4_t, aw.DType.f16)
    v_next = b.kv_cache_append(v_cache, v4_t, cache_write_index)

    # 4) Attention.
    attn_out = b.mha_cached(
        q=q,
        k=k_next,
        v=v_next,
        positions=positions,
        end_index=cache_visible_end,
        scale=1.0,
        sliding_window=sliding,
        softcap=0.0,
    )
    taps["attn_out"] = attn_out

    # 5) o_proj + residual + post-attn norm.
    attn_flat = b.reshape(attn_out, ("batch", "seq", NUM_HEADS * head_dim))
    o_out = b.matmul(attn_flat, lw.o_proj)
    o_out = b.rmsnorm(o_out, lw.post_attn_ln, b.zero_beta((EMBED_DIM,)), (EMBED_DIM,))
    taps["post_attn"] = o_out
    x2 = b.elemwise(aw.ElemwiseBinaryOp.add, x, o_out)
    taps["x_post_attn"] = x2

    # 6) FFN.
    ff_in = b.rmsnorm(x2, lw.pre_ffn_ln, b.zero_beta((EMBED_DIM,)), (EMBED_DIM,))
    gate = b.matmul(ff_in, lw.gate_proj)
    up = b.matmul(ff_in, lw.up_proj)
    gate_act = b.unary(aw.UnaryOp.gelu, gate)
    ff = b.elemwise(aw.ElemwiseBinaryOp.mul, gate_act, up)
    ff = b.matmul(ff, lw.down_proj)
    ff = b.rmsnorm(ff, lw.post_ffn_ln, b.zero_beta((EMBED_DIM,)), (EMBED_DIM,))
    taps["post_ffn"] = ff
    x3 = b.elemwise(aw.ElemwiseBinaryOp.add, x2, ff)
    taps["x_post_ffn"] = x3

    # 7) PLI mapping (per-layer input is a public input in this debug graph).
    pli_gate_in = b.matmul(x3, lw.pli_gate)
    pli_gate_act = b.unary(aw.UnaryOp.gelu, pli_gate_in)
    pli_combined = b.elemwise(aw.ElemwiseBinaryOp.mul, pli_gate_act, per_layer_input)
    pli_out = b.matmul(pli_combined, lw.pli_proj)
    pli_out = b.rmsnorm(pli_out, lw.post_pli_ln, b.zero_beta((EMBED_DIM,)), (EMBED_DIM,))
    taps["post_pli"] = pli_out
    x4 = b.elemwise(aw.ElemwiseBinaryOp.add, x3, pli_out)
    taps["x_post_pli"] = x4

    # 8) Skip scale.
    x_out = b.scalar_last_dim_mul(x4, EMBED_DIM, lw.skip_scale)
    taps["x_out"] = x_out
    return x_out, k_next, v_next, taps


def _reject_if_unexpected_multimodal(keys: List[str], allow: bool) -> None:
    if allow:
        return
    bad = [k for k in keys if k.startswith("model.audio_tower.") or k.startswith("model.vision_tower.")]
    if not bad:
        return
    sample = "\n  ".join(bad[:5])
    raise SystemExit(
        f"checkpoint contains {len(bad)} multimodal tensor(s); pass --allow-multimodal "
        f"to ignore them\ne.g.:\n  {sample}"
    )


class _WeightLoader:
    """safetensors reader that up-casts bf16 to f32 before handing values to the packer."""

    def __init__(self, path: str) -> None:
        self.path = path

    def __enter__(self) -> "_WeightLoader":
        from safetensors import safe_open
        self._file = safe_open(self.path, framework="pt")
        self._file.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._file.__exit__(exc_type, exc, tb)

    def keys(self) -> List[str]:
        return list(self._file.keys())

    def get_f32(self, name: str) -> np.ndarray:
        return self._file.get_tensor(name).float().numpy()


# -------------------------------- Quantization -------------------------------


def _as_f32_bytes(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr.astype(np.float32)).tobytes(order="C")


def _pack_matmul_weight_q8_0(w_torch_layout: np.ndarray) -> bytes:
    """PyTorch linear `[out, in]` → Aion matmul-B `[1, in, out]` packed as q8_0 quant_axis=1."""
    # Aion `MatMul` expects A and B to have the same rank. Gemma's residual stream
    # tensors are rank-3 `[B, S, K]`, so store dense weights as `[1, K, N]` and
    # broadcast across the batch dimension.
    w_t = np.ascontiguousarray(w_torch_layout.T)  # [K, N]
    if w_t.shape[0] % aw.Q8_0_BLOCK_ELEMS != 0:
        raise ValueError(f"matmul K={w_t.shape[0]} not a multiple of {aw.Q8_0_BLOCK_ELEMS}")
    w_t3 = w_t.reshape((1, int(w_t.shape[0]), int(w_t.shape[1])))
    return aw.pack_q8_0(w_t3, axis=1)


def _pack_embedding_q8_0(table: np.ndarray) -> bytes:
    if table.shape[1] % aw.Q8_0_BLOCK_ELEMS != 0:
        raise ValueError(f"embedding D={table.shape[1]} not a multiple of {aw.Q8_0_BLOCK_ELEMS}")
    return aw.pack_q8_0(np.ascontiguousarray(table), axis=1)


# ------------------------------ Package builder ------------------------------


Shape = Tuple[Union[int, str], ...]


class _PackageBuilder:
    """Holds `aw.Package` and exposes semantic helpers for values/nodes/initializers."""

    def __init__(self) -> None:
        self.pkg = aw.Package()
        for sym in ("batch", "seq", "G"):
            self.pkg.dim_symbols.append(sym)
            self.pkg.dim_exprs_symbol_indices.append(len(self.pkg.dim_exprs_symbol_indices))
        self._sym_idx: Dict[str, int] = {"batch": 0, "seq": 1, "G": 2}
        # Shared zero-beta initializers keyed by their normalized-shape tuple.
        self._zero_beta_by_shape: Dict[Tuple[int, ...], int] = {}
        # Shared scale-broadcast vectors keyed by (dim, value) → value id.
        self._scale_vec_by_key: Dict[Tuple[int, float], int] = {}
        # Rank bookkeeping: every value in the package must have rank > 0 per the on-disk
        # format validator. We track rank for produced values so each `_emit(...)` knows
        # what to stamp on the output.
        self._value_rank: Dict[int, int] = {}
        # DType bookkeeping: the on-disk package stores a concrete dtype for every value,
        # and signatures/IO-alias compatibility relies on it being correct.
        self._value_dtype: Dict[int, int] = {}

    # ---- value creators ----

    def _emit_shape_terms(self, shape: Shape) -> bytes:
        out = bytearray()
        out += aw.u32(len(shape))
        for dim in shape:
            if isinstance(dim, str):
                out += aw.shape_term_expr(self._sym_idx[dim])
            else:
                out += aw.shape_term_constant(int(dim))
        return bytes(out)

    def add_input(self, name: str, dtype: int, shape: Shape) -> int:
        vid = len(self.pkg.values)
        self.pkg.values.append(aw.ValueRecord(
            dtype=dtype,
            rank=len(shape),
            source=aw.ValueSource.public_input,
            shape_terms=self._emit_shape_terms(shape),
        ))
        self.pkg.inputs.append(aw.NamedValue(name=name, value=vid))
        self.pkg.debug_names.append(aw.DebugName(value=vid, name=name))
        self._value_rank[vid] = len(shape)
        self._value_dtype[vid] = dtype
        return vid

    def add_output(self, name: str, value_id: int) -> None:
        self.pkg.outputs.append(aw.NamedValue(name=name, value=value_id))

    def add_io_alias(self, input_index_in_signatures: int, output_index_in_signatures: int) -> None:
        self.pkg.io_aliases.append(aw.IoAlias(input_index_in_signatures, output_index_in_signatures))

    def _add_initializer_raw(
        self, name: str, init: aw.Initializer, value_dtype: int, shape: Tuple[int, ...]
    ) -> int:
        init_idx = len(self.pkg.initializers)
        self.pkg.initializers.append(init)
        vid = len(self.pkg.values)
        shape_bytes = bytearray(aw.u32(len(shape)))
        for d in shape:
            shape_bytes += aw.shape_term_constant(int(d))
        self.pkg.values.append(aw.ValueRecord(
            dtype=value_dtype,
            rank=len(shape),
            source=aw.ValueSource.initializer,
            shape_terms=bytes(shape_bytes),
            initializer_index=init_idx,
        ))
        self.pkg.debug_names.append(aw.DebugName(value=vid, name=name))
        self._value_rank[vid] = len(shape)
        self._value_dtype[vid] = value_dtype
        return vid

    def _dtype_of(self, value_id: int) -> int:
        return self._value_dtype[value_id]

    def add_f32_initializer(self, name: str, arr: np.ndarray) -> int:
        return self._add_initializer_raw(
            name,
            aw.Initializer.plain(aw.DType.f32, _as_f32_bytes(arr)),
            aw.DType.f32,
            tuple(int(x) for x in arr.shape),
        )

    def add_q8_0_matmul_b(self, name: str, w_torch: np.ndarray) -> int:
        packed = _pack_matmul_weight_q8_0(w_torch)
        in_dim, out_dim = int(w_torch.shape[1]), int(w_torch.shape[0])
        init = aw.Initializer.quantized_q8_0(
            logical_dtype=aw.DType.f16, quant_axis=1, data=packed,
        )
        return self._add_initializer_raw(name, init, aw.DType.q8_0, (1, in_dim, out_dim))

    def add_q8_0_embedding(self, name: str, table: np.ndarray) -> int:
        packed = _pack_embedding_q8_0(table)
        v, d = int(table.shape[0]), int(table.shape[1])
        init = aw.Initializer.quantized_q8_0(
            logical_dtype=aw.DType.f16, quant_axis=1, data=packed,
        )
        return self._add_initializer_raw(name, init, aw.DType.q8_0, (v, d))

    def zero_beta(self, shape: Tuple[int, ...]) -> int:
        key = tuple(int(x) for x in shape)
        if key not in self._zero_beta_by_shape:
            arr = np.zeros(key, dtype=np.float32)
            self._zero_beta_by_shape[key] = self.add_f32_initializer(
                f"_zero_beta_{'x'.join(str(x) for x in key)}", arr
            )
        return self._zero_beta_by_shape[key]

    def scale_broadcast_vec(self, dim: int, value: float) -> int:
        """A constant `[dim]` f32 vector of `value` used for last-dim scalar scaling."""
        key = (dim, float(value))
        if key not in self._scale_vec_by_key:
            arr = np.full((dim,), value, dtype=np.float32)
            self._scale_vec_by_key[key] = self.add_f32_initializer(
                f"_scale_{dim}_{value:g}", arr
            )
        return self._scale_vec_by_key[key]

    # ---- node emitters (produce a new value and return its id) ----

    def _add_produced_value(self, *, dtype: int, rank: int) -> int:
        if rank <= 0:
            raise ValueError(f"produced value must have rank >= 1 (got {rank})")
        vid = len(self.pkg.values)
        self.pkg.values.append(aw.ValueRecord(
            dtype=dtype,
            rank=rank,
            source=aw.ValueSource.produced,
            shape_terms=aw.u32(0),
        ))
        self._value_rank[vid] = rank
        self._value_dtype[vid] = dtype
        return vid

    def _emit(self, kind: int, inputs: List[int], attr: bytes, *, out_dtype: int, out_rank: int) -> int:
        out = self._add_produced_value(dtype=out_dtype, rank=out_rank)
        self.pkg.nodes.append(aw.NodeRecord(kind=kind, output=out, inputs=inputs, attr=attr))
        return out

    def matmul(self, a: int, b: int) -> int:
        # A[..., K] @ B[..., K, N] -> [..., N]; rank preserved from A.
        a_dt = self._dtype_of(a)
        b_dt = self._dtype_of(b)

        # Mirrors `src/aion/graph/infer.zig` dtype routing for MatMul.
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

        return self._emit(aw.NodeKind.MatMul, [a, b], aw.attr_matmul(), out_dtype=out_dt, out_rank=self._value_rank[a])

    def matmul_nt(self, a: int, b: int) -> int:
        # Mirrors `infer.zig:MatMulNT` (Gemma tied-logits path).
        if self._dtype_of(a) != aw.DType.f32:
            raise ValueError("MatMulNT: A must be f32")
        if self._dtype_of(b) != aw.DType.q8_0:
            raise ValueError("MatMulNT: B must be q8_0")
        return self._emit(aw.NodeKind.MatMulNT, [a, b], aw.attr_matmul_nt(), out_dtype=aw.DType.f32, out_rank=self._value_rank[a])

    def elemwise(self, op: int, a: int, b: int) -> int:
        a_dt = self._dtype_of(a)
        b_dt = self._dtype_of(b)
        if a_dt != b_dt:
            raise ValueError("ElemwiseBinary: dtype mismatch")
        if a_dt in (aw.DType.q4_0, aw.DType.q8_0):
            raise ValueError("ElemwiseBinary: quantized inputs are not supported")
        return self._emit(aw.NodeKind.ElemwiseBinary, [a, b], aw.attr_binary(op), out_dtype=a_dt, out_rank=self._value_rank[a])

    def broadcast_last_binary(self, op: int, a: int, b: int) -> int:
        a_dt = self._dtype_of(a)
        b_dt = self._dtype_of(b)
        if a_dt != b_dt:
            raise ValueError("BroadcastLastDimBinary: dtype mismatch")
        if a_dt in (aw.DType.q4_0, aw.DType.q8_0):
            raise ValueError("BroadcastLastDimBinary: quantized inputs are not supported")
        return self._emit(aw.NodeKind.BroadcastLastDimBinary, [a, b], aw.attr_binary(op), out_dtype=a_dt, out_rank=self._value_rank[a])

    def unary(self, op: int, a: int) -> int:
        a_dt = self._dtype_of(a)
        if a_dt in (aw.DType.q4_0, aw.DType.q8_0):
            raise ValueError("Unary: quantized inputs are not supported")
        return self._emit(aw.NodeKind.Unary, [a], aw.attr_unary(op), out_dtype=a_dt, out_rank=self._value_rank[a])

    def rmsnorm(self, x: int, gamma: int, beta: int, normalized_shape: Tuple[int, ...]) -> int:
        terms = [aw.shape_term_constant(int(d)) for d in normalized_shape]
        x_dt = self._dtype_of(x)
        if self._dtype_of(gamma) != x_dt or self._dtype_of(beta) != x_dt:
            raise ValueError("RMSNorm: dtype mismatch")
        if x_dt in (aw.DType.q4_0, aw.DType.q8_0):
            raise ValueError("RMSNorm: quantized inputs are not supported")
        return self._emit(aw.NodeKind.RMSNorm, [x, gamma, beta], aw.attr_rmsnorm(RMS_EPS, terms), out_dtype=x_dt, out_rank=self._value_rank[x])

    def reshape(self, x: int, new_shape: Shape) -> int:
        terms = []
        for dim in new_shape:
            if isinstance(dim, str):
                terms.append(aw.shape_term_expr(self._sym_idx[dim]))
            else:
                terms.append(aw.shape_term_constant(int(dim)))
        return self._emit(aw.NodeKind.ViewReshape, [x], aw.attr_view_reshape(terms), out_dtype=self._dtype_of(x), out_rank=len(new_shape))

    def slice_nd(self, x: int, starts: Tuple[int, ...], lens: Shape) -> int:
        lens_terms = []
        for dim in lens:
            if isinstance(dim, str):
                lens_terms.append(aw.shape_term_expr(self._sym_idx[dim]))
            else:
                lens_terms.append(aw.shape_term_constant(int(dim)))
        return self._emit(
            aw.NodeKind.ViewSliceND,
            [x],
            aw.attr_view_slice_nd(starts, lens_terms),
            out_dtype=self._dtype_of(x),
            out_rank=len(lens),
        )

    def rope1d(
        self,
        x: int,
        positions: int,
        base: float,
        *,
        scale_factor: float = 1.0,
        rope_proportion: float = 1.0,
    ) -> int:
        return self._emit(
            aw.NodeKind.RoPE1D,
            [x, positions],
            aw.attr_rope1d(base_frequency=base, scale_factor=scale_factor, rope_proportion=rope_proportion),
            out_dtype=self._dtype_of(x),
            out_rank=self._value_rank[x],
        )

    def cast(self, x: int, to_dtype: int) -> int:
        x_dt = self._dtype_of(x)
        if x_dt in (aw.DType.q4_0, aw.DType.q8_0) or to_dtype in (aw.DType.q4_0, aw.DType.q8_0):
            raise ValueError("Cast: quantized casts are not supported")
        ok = (
            (x_dt == to_dtype)
            or (x_dt == aw.DType.f16 and to_dtype == aw.DType.f32)
            or (x_dt == aw.DType.f32 and to_dtype == aw.DType.f16)
        )
        if not ok:
            raise ValueError(f"Cast: unsupported cast {x_dt} -> {to_dtype}")
        return self._emit(aw.NodeKind.Cast, [x], aw.attr_cast(to_dtype), out_dtype=to_dtype, out_rank=self._value_rank[x])

    def kv_cache_append(self, cache: int, new_kv: int, end_index: int) -> int:
        cache_dt = self._dtype_of(cache)
        if self._dtype_of(new_kv) != cache_dt:
            raise ValueError("KVCacheAppend: cache/new_kv dtype mismatch")
        if cache_dt not in (aw.DType.f16, aw.DType.f32):
            raise ValueError("KVCacheAppend: unsupported cache dtype")
        if self._dtype_of(end_index) != aw.DType.i32:
            raise ValueError("KVCacheAppend: end_index must be i32")
        return self._emit(aw.NodeKind.KVCacheAppend, [cache, new_kv, end_index], b"", out_dtype=cache_dt, out_rank=self._value_rank[cache])

    def gather_rows(self, table: int, indices: int) -> int:
        table_dt = self._dtype_of(table)
        if self._dtype_of(indices) != aw.DType.i32:
            raise ValueError("GatherRows: indices must be i32")
        if table_dt in (aw.DType.f16, aw.DType.f32):
            out_dt = table_dt
        elif table_dt == aw.DType.q8_0:
            # Mirrors `infer.zig`: q8_0 table materializes f32 output by default.
            out_dt = aw.DType.f32
        else:
            raise ValueError("GatherRows: unsupported table dtype")
        return self._emit(aw.NodeKind.GatherRows, [table, indices], b"", out_dtype=out_dt, out_rank=3)

    def mha_cached(
        self,
        q: int,
        k: int,
        v: int,
        positions: int,
        end_index: int,
        scale: float,
        sliding_window: int,
        softcap: float,
    ) -> int:
        return self._emit(
            aw.NodeKind.MultiHeadAttentionCached,
            [q, k, v, positions, end_index],
            aw.attr_mha_cached(scale=scale, causal=True, sliding_window=sliding_window, attn_logits_soft_cap=softcap),
            out_dtype=aw.DType.f32,
            out_rank=self._value_rank[q],
        )

    # ---- convenience: scalar broadcast multiply on [..., D] via a [D] constant ----

    def scalar_last_dim_mul(self, x: int, dim: int, value: float) -> int:
        vec = self.scale_broadcast_vec(dim, value)
        return self.broadcast_last_binary(aw.ElemwiseBinaryOp.mul, x, vec)

    def scalar_last_dim_div(self, x: int, dim: int, value: float) -> int:
        vec = self.scale_broadcast_vec(dim, value)
        return self.broadcast_last_binary(aw.ElemwiseBinaryOp.div, x, vec)


# ------------------------------- Forward pass --------------------------------


@dataclass
class _LayerWeights:
    input_ln: int
    post_attn_ln: int
    pre_ffn_ln: int
    post_ffn_ln: int
    post_pli_ln: int
    skip_scale: float

    o_proj: int
    q_norm: int

    k_proj: Optional[int]
    v_proj: Optional[int]
    k_norm: Optional[int]

    down_proj: int

    pli_gate: int
    pli_proj: int

    # Per-layer FFN width. Gemma 4 E2B is elastic: layers 0..14 use FFN=6144, layers
    # 15..34 use FFN=12288. The fused gate/up split point must therefore be per-layer,
    # NOT the FFN_DIM constant.
    ffn_dim: int = FFN_DIM

    # Attention Q projection: non-source layers use a standalone `q_proj`; source layers
    # (0..14, which also compute K/V) fuse Q+K+V into one matmul-B split by slices, same
    # rationale as the gate/up fusion below.
    q_proj: Optional[int] = None
    qkv_proj: Optional[int] = None

    # FFN gate/up: either stored separately (debug single-block path) or fused into a
    # single [in, 2*ffn_dim] matmul-B (main path) and split with two slices in the graph.
    # Fusing halves the decode op count for the FFN's biggest matmul and lets the GEMV
    # amortize per-op overhead (~1.9x faster on the gate/up step; see `zig build bench
    # -- --suite decode`). Defaulted so each emit path sets only what it uses.
    gate_proj: Optional[int] = None
    up_proj: Optional[int] = None
    gateup_proj: Optional[int] = None


@dataclass
class _SharedWeights:
    embed_tokens: int
    embed_tokens_per_layer: int
    per_layer_model_projection: int
    per_layer_projection_norm: int
    final_norm: int


def _emit_weights(loader: _WeightLoader, b: _PackageBuilder) -> Tuple[_SharedWeights, List[_LayerWeights]]:
    ln = "model.language_model"

    def f32v(name: str, arr: np.ndarray) -> int:
        return b.add_f32_initializer(name, arr)

    shared = _SharedWeights(
        embed_tokens=b.add_q8_0_embedding("embed_tokens.weight", loader.get_f32(f"{ln}.embed_tokens.weight")),
        embed_tokens_per_layer=b.add_q8_0_embedding(
            "embed_tokens_per_layer.weight", loader.get_f32(f"{ln}.embed_tokens_per_layer.weight")
        ),
        per_layer_model_projection=b.add_q8_0_matmul_b(
            "per_layer_model_projection.weight", loader.get_f32(f"{ln}.per_layer_model_projection.weight")
        ),
        per_layer_projection_norm=f32v(
            "per_layer_projection_norm.weight", loader.get_f32(f"{ln}.per_layer_projection_norm.weight")
        ),
        final_norm=f32v("norm.weight", loader.get_f32(f"{ln}.norm.weight")),
    )

    layers: List[_LayerWeights] = []
    for layer in range(NUM_LAYERS):
        pfx = f"{ln}.layers.{layer}"
        name = lambda leaf: f"layers.{layer}.{leaf}"

        is_source = layer in SOURCE_LAYERS

        # Elastic FFN: per-layer width is the gate_proj output dim (6144 or 12288).
        gate_w = loader.get_f32(f"{pfx}.mlp.gate_proj.weight")
        up_w = loader.get_f32(f"{pfx}.mlp.up_proj.weight")
        layer_ffn = int(gate_w.shape[0])

        lw = _LayerWeights(
            input_ln=f32v(name("input_layernorm.weight"), loader.get_f32(f"{pfx}.input_layernorm.weight")),
            post_attn_ln=f32v(name("post_attention_layernorm.weight"), loader.get_f32(f"{pfx}.post_attention_layernorm.weight")),
            pre_ffn_ln=f32v(name("pre_feedforward_layernorm.weight"), loader.get_f32(f"{pfx}.pre_feedforward_layernorm.weight")),
            post_ffn_ln=f32v(name("post_feedforward_layernorm.weight"), loader.get_f32(f"{pfx}.post_feedforward_layernorm.weight")),
            post_pli_ln=f32v(name("post_per_layer_input_norm.weight"), loader.get_f32(f"{pfx}.post_per_layer_input_norm.weight")),
            skip_scale=float(loader.get_f32(f"{pfx}.layer_scalar").reshape(-1)[0]),
            o_proj=b.add_q8_0_matmul_b(name("self_attn.o_proj.weight"), loader.get_f32(f"{pfx}.self_attn.o_proj.weight")),
            q_norm=f32v(name("self_attn.q_norm.weight"), loader.get_f32(f"{pfx}.self_attn.q_norm.weight")),
            k_proj=None, v_proj=None, k_norm=None,
            # Fuse gate_proj+up_proj into one [2*ffn, in] weight (PyTorch [out, in]),
            # concatenated along the output axis: rows [0:ffn]=gate, [ffn:]=up.
            ffn_dim=layer_ffn,
            gateup_proj=b.add_q8_0_matmul_b(
                name("mlp.gateup_proj.weight"),
                np.concatenate([gate_w, up_w], axis=0),
            ),
            down_proj=b.add_q8_0_matmul_b(name("mlp.down_proj.weight"), loader.get_f32(f"{pfx}.mlp.down_proj.weight")),
            pli_gate=b.add_q8_0_matmul_b(name("per_layer_input_gate.weight"), loader.get_f32(f"{pfx}.per_layer_input_gate.weight")),
            pli_proj=b.add_q8_0_matmul_b(name("per_layer_projection.weight"), loader.get_f32(f"{pfx}.per_layer_projection.weight")),
        )
        if is_source:
            # Source layers fuse Q+K+V into one matmul-B (rows: Q, then K, then V),
            # split back with slices in the forward graph.
            lw.qkv_proj = b.add_q8_0_matmul_b(
                name("self_attn.qkv_proj.weight"),
                np.concatenate(
                    [
                        loader.get_f32(f"{pfx}.self_attn.q_proj.weight"),
                        loader.get_f32(f"{pfx}.self_attn.k_proj.weight"),
                        loader.get_f32(f"{pfx}.self_attn.v_proj.weight"),
                    ],
                    axis=0,
                ),
            )
            lw.k_norm = f32v(name("self_attn.k_norm.weight"), loader.get_f32(f"{pfx}.self_attn.k_norm.weight"))
        else:
            lw.q_proj = b.add_q8_0_matmul_b(name("self_attn.q_proj.weight"), loader.get_f32(f"{pfx}.self_attn.q_proj.weight"))
        layers.append(lw)

    return shared, layers


def _emit_forward(
    b: _PackageBuilder,
    loader_keys_log: List[str],
    shared: _SharedWeights,
    layers: List[_LayerWeights],
) -> Tuple[int, Dict[int, Tuple[int, int]]]:
    """Emit the full forward graph. Returns (logits_vid, per-source-layer (K,V) append outputs)."""

    # Public runtime inputs.
    tokens = b.add_input("tokens", aw.DType.i32, ("batch", "seq"))
    positions = b.add_input("positions", aw.DType.i32, ("batch", "seq"))
    cache_write_index = b.add_input("cache_write_index", aw.DType.i32, ("batch",))
    cache_visible_end = b.add_input("cache_visible_end", aw.DType.i32, ("batch",))
    loader_keys_log.extend(["tokens", "positions", "cache_write_index", "cache_visible_end"])

    # KV cache public inputs — f16, one pair per source layer.
    k_cache_in: Dict[int, int] = {}
    v_cache_in: Dict[int, int] = {}
    for src in SOURCE_LAYERS:
        is_glob = is_global_layer(src)
        head_dim = GLOBAL_HEAD_DIM if is_glob else LOCAL_HEAD_DIM
        t_dim: Union[int, str] = "G" if is_glob else LOCAL_SLIDING_WINDOW
        k_cache_in[src] = b.add_input(f"k_cache.layer{src}", aw.DType.f16, ("batch", NUM_KV_HEADS, t_dim, head_dim))
        v_cache_in[src] = b.add_input(f"v_cache.layer{src}", aw.DType.f16, ("batch", NUM_KV_HEADS, t_dim, head_dim))

    # ---- Embedding + PLI encoder ----

    # 1) Token embedding lookup; output is f32 [B, S, 1536].
    emb = b.gather_rows(shared.embed_tokens, tokens)
    # 2) Scale by sqrt(D).
    emb_scaled = b.scalar_last_dim_mul(emb, EMBED_DIM, math.sqrt(EMBED_DIM))

    # 3) PLI projection path: proj = emb_scaled @ per_layer_model_projection.T, scale, reshape, RMSNorm.
    #    The PyTorch weight is [8960, 1536]; stored as matmul-B [1, 1536, 8960] quant_axis=1.
    pli_proj_flat = b.matmul(emb_scaled, shared.per_layer_model_projection)  # [B, S, 8960]
    pli_proj_scaled = b.scalar_last_dim_mul(pli_proj_flat, PLI_TOTAL, 1.0 / math.sqrt(EMBED_DIM))
    pli_proj_r = b.reshape(pli_proj_scaled, ("batch", "seq", NUM_LAYERS, PLI_DIM))
    pli_proj_norm = b.rmsnorm(pli_proj_r, shared.per_layer_projection_norm, b.zero_beta((PLI_DIM,)), (PLI_DIM,))

    # 4) PLI embedding path: gather pli_table, scale by sqrt(D_pli), reshape.
    pli_emb_flat = b.gather_rows(shared.embed_tokens_per_layer, tokens)  # [B,S,8960]
    pli_emb_scaled = b.scalar_last_dim_mul(pli_emb_flat, PLI_TOTAL, math.sqrt(PLI_DIM))
    pli_emb_r = b.reshape(pli_emb_scaled, ("batch", "seq", NUM_LAYERS, PLI_DIM))

    # 5) Sum-then-scale.
    pli_sum = b.elemwise(aw.ElemwiseBinaryOp.add, pli_proj_norm, pli_emb_r)
    pli = b.scalar_last_dim_mul(pli_sum, PLI_DIM, 1.0 / math.sqrt(2.0))  # [B,S,35,256]

    # Residual stream.
    x = emb_scaled

    # Track the latest cache-post-append value id per source layer.
    k_cache_cur: Dict[int, int] = dict(k_cache_in)
    v_cache_cur: Dict[int, int] = dict(v_cache_in)

    # ---- Per-layer transformer blocks ----

    for layer_idx, lw in enumerate(layers):
        is_glob = is_global_layer(layer_idx)
        head_dim = GLOBAL_HEAD_DIM if is_glob else LOCAL_HEAD_DIM
        sliding = 0 if is_glob else LOCAL_SLIDING_WINDOW
        rope_base = ROPE_GLOBAL_BASE if is_glob else ROPE_LOCAL_BASE
        rope_prop = ROPE_GLOBAL_PROPORTION if is_glob else ROPE_LOCAL_PROPORTION
        src = kv_source_of(layer_idx)
        is_source = layer_idx in SOURCE_LAYERS

        # 1) Pre-attn norm on x.
        x_norm = b.rmsnorm(x, lw.input_ln, b.zero_beta((EMBED_DIM,)), (EMBED_DIM,))

        # 2) Q projection (non-source) or fused QKV projection split by slices (source).
        q_out = NUM_HEADS * head_dim
        kv_out = NUM_KV_HEADS * head_dim
        if is_source:
            assert lw.qkv_proj is not None
            qkv = b.matmul(x_norm, lw.qkv_proj)  # [B, S, q_out + 2*kv_out]
            q_flat = b.slice_nd(qkv, (0, 0, 0), ("batch", "seq", q_out))
            k_flat = b.slice_nd(qkv, (0, 0, q_out), ("batch", "seq", kv_out))
            v_flat = b.slice_nd(qkv, (0, 0, q_out + kv_out), ("batch", "seq", kv_out))
        else:
            q_flat = b.matmul(x_norm, lw.q_proj)  # [B, S, H*D_k]

        q = b.reshape(q_flat, ("batch", "seq", NUM_HEADS, head_dim))
        q = b.rmsnorm(q, lw.q_norm, b.zero_beta((head_dim,)), (head_dim,))
        q = b.rope1d(q, positions, rope_base, rope_proportion=rope_prop)

        # 3) For source layers: norm/rope/append the K/V slices (after cast to f16).
        if is_source:
            assert lw.k_norm is not None
            k4 = b.reshape(k_flat, ("batch", "seq", NUM_KV_HEADS, head_dim))
            k4 = b.rmsnorm(k4, lw.k_norm, b.zero_beta((head_dim,)), (head_dim,))
            k4 = b.rope1d(k4, positions, rope_base, rope_proportion=rope_prop)
            # Append semantics want [B, H_kv, new_len, D_k]; we have [B, seq, H_kv, D_k].
            # The runtime append kernel expects shape[0]=B, shape[1]=H_kv, shape[2]=new_len, shape[3]=D_k.
            k4_t = b.reshape(k4, ("batch", NUM_KV_HEADS, "seq", head_dim))
            k_f16 = b.cast(k4_t, aw.DType.f16)
            k_cache_cur[layer_idx] = b.kv_cache_append(k_cache_cur[layer_idx], k_f16, cache_write_index)

            v4 = b.reshape(v_flat, ("batch", "seq", NUM_KV_HEADS, head_dim))
            # Gemma4 uses V-norm, but it is parameterless (gamma is implicitly all-ones).
            v4 = b.rmsnorm(v4, b.scale_broadcast_vec(head_dim, 1.0), b.zero_beta((head_dim,)), (head_dim,))
            v4_t = b.reshape(v4, ("batch", NUM_KV_HEADS, "seq", head_dim))
            v_f16 = b.cast(v4_t, aw.DType.f16)
            v_cache_cur[layer_idx] = b.kv_cache_append(v_cache_cur[layer_idx], v_f16, cache_write_index)

        # 4) Attention reads the current (post-append) cache of the source layer.
        k_ref = k_cache_cur[src]
        v_ref = v_cache_cur[src]
        attn_out = b.mha_cached(
            q=q,
            k=k_ref,
            v=v_ref,
            positions=positions,
            end_index=cache_visible_end,
            # QK-norm replaces 1/sqrt(d) scaling in Gemma4; attention scale is 1.0.
            scale=1.0,
            sliding_window=sliding,
            softcap=0.0,  # attn-logit softcap; Gemma E2B uses 0 per plan's defaults.
        )

        # 5) Reshape + o_proj + residual + post-attn norm.
        attn_flat = b.reshape(attn_out, ("batch", "seq", NUM_HEADS * head_dim))
        o_out = b.matmul(attn_flat, lw.o_proj)
        o_out = b.rmsnorm(o_out, lw.post_attn_ln, b.zero_beta((EMBED_DIM,)), (EMBED_DIM,))
        x = b.elemwise(aw.ElemwiseBinaryOp.add, x, o_out)

        # 6) FFN: pre-ffn norm, fused gate/up matmul (split via slices), gelu, multiply,
        #    down, residual, post-ffn norm.
        ff_in = b.rmsnorm(x, lw.pre_ffn_ln, b.zero_beta((EMBED_DIM,)), (EMBED_DIM,))
        ffn = lw.ffn_dim                               # per-layer (elastic): 6144 or 12288
        gate_up = b.matmul(ff_in, lw.gateup_proj)      # [B, S, 2*ffn]
        gate = b.slice_nd(gate_up, (0, 0, 0), ("batch", "seq", ffn))    # rows [0:ffn]
        up = b.slice_nd(gate_up, (0, 0, ffn), ("batch", "seq", ffn))    # rows [ffn:]
        gate_act = b.unary(aw.UnaryOp.gelu, gate)
        ff = b.elemwise(aw.ElemwiseBinaryOp.mul, gate_act, up)
        ff = b.matmul(ff, lw.down_proj)                # [B, S, EMBED_DIM]
        ff = b.rmsnorm(ff, lw.post_ffn_ln, b.zero_beta((EMBED_DIM,)), (EMBED_DIM,))
        x = b.elemwise(aw.ElemwiseBinaryOp.add, x, ff)

        # 7) PLI mapping.
        pli_gate_in = b.matmul(x, lw.pli_gate)                   # [B, S, PLI_DIM]
        pli_gate_act = b.unary(aw.UnaryOp.gelu, pli_gate_in)
        pli_slice = b.slice_nd(pli, (0, 0, layer_idx, 0), ("batch", "seq", 1, PLI_DIM))
        pli_slice2 = b.reshape(pli_slice, ("batch", "seq", PLI_DIM))
        pli_combined = b.elemwise(aw.ElemwiseBinaryOp.mul, pli_gate_act, pli_slice2)
        pli_out = b.matmul(pli_combined, lw.pli_proj)            # [B, S, EMBED_DIM]
        pli_out = b.rmsnorm(pli_out, lw.post_pli_ln, b.zero_beta((EMBED_DIM,)), (EMBED_DIM,))
        x = b.elemwise(aw.ElemwiseBinaryOp.add, x, pli_out)

        # 8) Skip scale.
        x = b.scalar_last_dim_mul(x, EMBED_DIM, lw.skip_scale)

    # ---- Tail ----

    # Final RMSNorm.
    x = b.rmsnorm(x, shared.final_norm, b.zero_beta((EMBED_DIM,)), (EMBED_DIM,))

    # Tied logits via MatMulNT: [B, S, V] = x[B, S, EMBED_DIM] @ embed_tokens[V, EMBED_DIM]^T
    logits = b.matmul_nt(x, shared.embed_tokens)

    # Softcap: tanh(x / 30) * 30 using last-dim broadcast vectors of size V.
    logits = b.scalar_last_dim_div(logits, VOCAB_SIZE, FINAL_LOGIT_SOFTCAP)
    logits = b.unary(aw.UnaryOp.tanh, logits)
    logits = b.scalar_last_dim_mul(logits, VOCAB_SIZE, FINAL_LOGIT_SOFTCAP)

    # Record (k_append, v_append) per source layer for output aliasing.
    source_append_outputs: Dict[int, Tuple[int, int]] = {
        src: (k_cache_cur[src], v_cache_cur[src]) for src in SOURCE_LAYERS
    }

    return logits, source_append_outputs


def _write_metadata(b: _PackageBuilder) -> None:
    b.pkg.metadata.extend([
        aw.MetadataEntry(key="arch", value="gemma4-e2b-text"),
        aw.MetadataEntry(key="quant", value="q8_0-weights-f16-cache"),
        aw.MetadataEntry(key="num_layers", value=str(NUM_LAYERS)),
        aw.MetadataEntry(key="embed_dim", value=str(EMBED_DIM)),
        aw.MetadataEntry(key="local_head_dim", value=str(LOCAL_HEAD_DIM)),
        aw.MetadataEntry(key="global_head_dim", value=str(GLOBAL_HEAD_DIM)),
        aw.MetadataEntry(key="local_sliding_window", value=str(LOCAL_SLIDING_WINDOW)),
        aw.MetadataEntry(key="final_logit_softcap", value=str(FINAL_LOGIT_SOFTCAP)),
        aw.MetadataEntry(key="kv_sharing.local_source_layer", value=str(LOCAL_SOURCE_LAYER)),
        aw.MetadataEntry(key="kv_sharing.global_source_layer", value=str(GLOBAL_SOURCE_LAYER)),
        aw.MetadataEntry(key="graph_phase", value="2-full-forward"),
    ])


def _alias_cache_outputs(
    b: _PackageBuilder,
    source_append_outputs: Dict[int, Tuple[int, int]],
) -> None:
    """Add `next_k_cache.layerX` / `next_v_cache.layerX` outputs and alias them back to the
    corresponding K/V cache inputs. The runtime relies on this to write back cache state."""
    name_to_in_idx = {nv.name: idx for idx, nv in enumerate(b.pkg.inputs)}
    for src, (k_out_vid, v_out_vid) in source_append_outputs.items():
        out_k_name = f"next_k_cache.layer{src}"
        out_v_name = f"next_v_cache.layer{src}"
        k_out_idx = len(b.pkg.outputs)
        b.add_output(out_k_name, k_out_vid)
        v_out_idx = len(b.pkg.outputs)
        b.add_output(out_v_name, v_out_vid)
        b.add_io_alias(name_to_in_idx[f"k_cache.layer{src}"], k_out_idx)
        b.add_io_alias(name_to_in_idx[f"v_cache.layer{src}"], v_out_idx)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    b = _PackageBuilder()
    log: List[str] = []

    with _WeightLoader(args.in_safetensors) as loader:
        _reject_if_unexpected_multimodal(loader.keys(), args.allow_multimodal)
        if args.debug_block_layer is not None:
            cache_dtype: int = aw.DType.f16 if args.debug_block_cache_dtype == "f16" else aw.DType.f32
            lw = _emit_one_layer_weights(loader, b, args.debug_block_layer, weight_mode=args.debug_block_weights)
        else:
            shared, layers = _emit_weights(loader, b)

    if args.debug_block_layer is not None:
        x_out, k_next, v_next, taps = _emit_forward_one_block(
            b,
            layer_idx=args.debug_block_layer,
            lw=lw,
            cache_dtype=cache_dtype,
        )
        b.add_output("x_out", x_out)
        k_out_idx = len(b.pkg.outputs)
        b.add_output("next_k_cache", k_next)
        v_out_idx = len(b.pkg.outputs)
        b.add_output("next_v_cache", v_next)

        name_to_in_idx = {nv.name: idx for idx, nv in enumerate(b.pkg.inputs)}
        b.add_io_alias(name_to_in_idx["k_cache"], k_out_idx)
        b.add_io_alias(name_to_in_idx["v_cache"], v_out_idx)

        if args.debug_block_taps:
            # Keep names stable so downstream scripts can compare step-by-step.
            for k in sorted(taps.keys()):
                if k in {"x_out"}:
                    continue
                b.add_output(f"tap.{k}", taps[k])

        b.pkg.metadata.extend([
            aw.MetadataEntry(key="arch", value="gemma4-e2b-text"),
            aw.MetadataEntry(key="graph_phase", value="debug-single-block"),
            aw.MetadataEntry(key="debug.layer", value=str(args.debug_block_layer)),
            aw.MetadataEntry(key="debug.weights", value=str(args.debug_block_weights)),
            aw.MetadataEntry(key="debug.cache_dtype", value=str(args.debug_block_cache_dtype)),
            aw.MetadataEntry(key="debug.taps", value=str(bool(args.debug_block_taps))),
        ])
    else:
        logits, source_append_outputs = _emit_forward(b, log, shared, layers)

        # Primary model output.
        b.add_output("logits", logits)
        # next_*_cache outputs + io_aliases.
        _alias_cache_outputs(b, source_append_outputs)

        _write_metadata(b)

    total = sum(len(init.data) for init in b.pkg.initializers)
    print(
        f"[gemma4-e2b] initializers: {len(b.pkg.initializers)}   "
        f"nodes: {len(b.pkg.nodes)}   "
        f"total initializer bytes: {total/1e9:.3f} GB"
    )

    # Sanity: `io_aliases` must always be compatible. If they are not, the runtime has
    # to ignore them (or error), and we lose the intended cache write-back behavior.
    for alias in b.pkg.io_aliases:
        if not (0 <= alias.input_index < len(b.pkg.inputs)):
            raise AssertionError(f"io_alias input_index out of range: {alias}")
        if not (0 <= alias.output_index < len(b.pkg.outputs)):
            raise AssertionError(f"io_alias output_index out of range: {alias}")

        in_nv = b.pkg.inputs[alias.input_index]
        out_nv = b.pkg.outputs[alias.output_index]
        in_v = b.pkg.values[in_nv.value]
        out_v = b.pkg.values[out_nv.value]
        if in_v.dtype != out_v.dtype or in_v.rank != out_v.rank:
            raise AssertionError(
                "io_alias incompatible: "
                f"input[{alias.input_index}]={in_nv.name} (vid={in_nv.value} dtype={in_v.dtype} rank={in_v.rank}) "
                f"-> output[{alias.output_index}]={out_nv.name} (vid={out_nv.value} dtype={out_v.dtype} rank={out_v.rank})"
            )

    aw.write_aion_v4(args.out_aion, b.pkg)
    print(f"[gemma4-e2b] wrote {args.out_aion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
