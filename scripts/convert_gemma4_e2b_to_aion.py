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
"""Convert Gemma 4 E2B (text-only) safetensors -> a self-contained q8_0 `.aion`.

Emits ONE full-forward model: token embedding + per-layer-input encoder + 35
transformer blocks (KV-cache attention with local/global rope, elastic FFN, PLI)
+ tied-embedding logits with final softcap. Authored via the C-ABI `aion.Builder`;
the Zig core serializes through `Builder.export()` (no pure-Python writer).

Weight/quant policy:
- dense matmul weights -> q8_0 stored matmul-B `[1, K, N]` (blocked on K).
- embedding tables -> q8_0 blocked along the feature dim.
- LayerNorm/RMSNorm gammas, per-layer scalars -> f32.
- KV caches are f16; global-layer caches use the growable capacity symbol `G`.

Run:  uv run --project bindings/python \
        python scripts/convert_gemma4_e2b_to_aion.py <in.safetensors> <out.aion>
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import aion
from aion import Builder, Value
from aion.enums import AionInputRoleKind as RK


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


# Authoring placeholder sizes for the symbolic axes (eager inference only; the
# exported model serves any runtime size on them).
_PLACEHOLDER = {"batch": 1, "seq": 1, "G": LOCAL_SLIDING_WINDOW}

Shape = Tuple[Union[int, str], ...]


# ------------------------------- CLI + loading -------------------------------


def _parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert Gemma 4 E2B text-only to a q8_0 .aion")
    ap.add_argument("in_safetensors", type=str)
    ap.add_argument("out_aion", type=str)
    ap.add_argument(
        "--allow-multimodal",
        action="store_true",
        help="Ignore audio/vision tensors in a multimodal checkpoint (they are not included).",
    )
    return ap.parse_args(argv)


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


# --------------------- C-ABI Builder authoring helpers -----------------------


def _q8b(b: Builder, w_torch: np.ndarray) -> Value:
    """PyTorch linear `[out, in]` -> q8_0 matmul-B `[1, in, out]` (blocked on K=in)."""
    w_t = np.ascontiguousarray(w_torch.astype(np.float32, copy=False).T)  # [K, N]
    k, n = int(w_t.shape[0]), int(w_t.shape[1])
    return b.param(w_t.reshape(1, k, n), dtype="q8_0", shape=(1, k, n))


def _q8_embed(b: Builder, table: np.ndarray) -> Value:
    """Embedding table `[V, D]` -> q8_0 blocked along the feature dim D (axis 1)."""
    t = np.ascontiguousarray(table.astype(np.float32, copy=False))
    v, d = int(t.shape[0]), int(t.shape[1])
    return b.param(t, dtype="q8_0", shape=(v, d), quant_axis=1)


def _f32(b: Builder, arr: np.ndarray) -> Value:
    return b.param(np.ascontiguousarray(arr.astype(np.float32, copy=False)))


def _input(b: Builder, name: str, dtype: str, shape: Shape) -> Value:
    """Declare a public input; string dims are symbolic (dynamic) axes."""
    dims = [_PLACEHOLDER[d] if isinstance(d, str) else int(d) for d in shape]
    dyn = {i: d for i, d in enumerate(shape) if isinstance(d, str)} or None
    return b.input(dims, dtype=dtype, dynamic=dyn).rename(name)


class _Consts:
    """Per-conversion cache of tiny constant params (zero betas, scalar vectors)."""

    def __init__(self, b: Builder) -> None:
        self.b = b
        self._zero: Dict[int, Value] = {}
        self._vec: Dict[Tuple[int, float], Value] = {}

    def zero_beta(self, dim: int) -> Value:
        v = self._zero.get(dim)
        if v is None:
            v = self.b.param(np.zeros((dim,), np.float32))
            self._zero[dim] = v
        return v

    def scale_vec(self, dim: int, value: float) -> Value:
        key = (dim, float(value))
        v = self._vec.get(key)
        if v is None:
            v = self.b.param(np.full((dim,), value, np.float32))
            self._vec[key] = v
        return v


def _rmsnorm(b: Builder, c: _Consts, x: Value, gamma: Value, dim: int) -> Value:
    return b.rmsnorm(x, gamma, c.zero_beta(dim), eps=RMS_EPS, normalized_shape=(dim,))


def _scalar_mul(b: Builder, c: _Consts, x: Value, dim: int, value: float) -> Value:
    return b.broadcast_mul(x, c.scale_vec(dim, value))


def _scalar_div(b: Builder, c: _Consts, x: Value, dim: int, value: float) -> Value:
    return b.broadcast_div(x, c.scale_vec(dim, value))


# ------------------------------- weight structs ------------------------------


@dataclass
class _LayerWeights:
    input_ln: Value
    post_attn_ln: Value
    pre_ffn_ln: Value
    post_ffn_ln: Value
    post_pli_ln: Value
    skip_scale: float
    o_proj: Value
    q_norm: Value
    down_proj: Value
    pli_gate: Value
    pli_proj: Value
    ffn_dim: int = FFN_DIM
    q_proj: Optional[Value] = None
    qkv_proj: Optional[Value] = None
    k_norm: Optional[Value] = None
    gateup_proj: Optional[Value] = None


@dataclass
class _SharedWeights:
    embed_tokens: Value
    embed_tokens_per_layer: Value
    per_layer_model_projection: Value
    per_layer_projection_norm: Value
    final_norm: Value


def _emit_weights(loader: _WeightLoader, b: Builder) -> Tuple[_SharedWeights, List[_LayerWeights]]:
    ln = "model.language_model"

    shared = _SharedWeights(
        embed_tokens=_q8_embed(b, loader.get_f32(f"{ln}.embed_tokens.weight")),
        embed_tokens_per_layer=_q8_embed(b, loader.get_f32(f"{ln}.embed_tokens_per_layer.weight")),
        per_layer_model_projection=_q8b(b, loader.get_f32(f"{ln}.per_layer_model_projection.weight")),
        per_layer_projection_norm=_f32(b, loader.get_f32(f"{ln}.per_layer_projection_norm.weight")),
        final_norm=_f32(b, loader.get_f32(f"{ln}.norm.weight")),
    )

    layers: List[_LayerWeights] = []
    for layer in range(NUM_LAYERS):
        pfx = f"{ln}.layers.{layer}"
        is_source = layer in SOURCE_LAYERS

        gate_w = loader.get_f32(f"{pfx}.mlp.gate_proj.weight")
        up_w = loader.get_f32(f"{pfx}.mlp.up_proj.weight")
        layer_ffn = int(gate_w.shape[0])

        lw = _LayerWeights(
            input_ln=_f32(b, loader.get_f32(f"{pfx}.input_layernorm.weight")),
            post_attn_ln=_f32(b, loader.get_f32(f"{pfx}.post_attention_layernorm.weight")),
            pre_ffn_ln=_f32(b, loader.get_f32(f"{pfx}.pre_feedforward_layernorm.weight")),
            post_ffn_ln=_f32(b, loader.get_f32(f"{pfx}.post_feedforward_layernorm.weight")),
            post_pli_ln=_f32(b, loader.get_f32(f"{pfx}.post_per_layer_input_norm.weight")),
            skip_scale=float(loader.get_f32(f"{pfx}.layer_scalar").reshape(-1)[0]),
            o_proj=_q8b(b, loader.get_f32(f"{pfx}.self_attn.o_proj.weight")),
            q_norm=_f32(b, loader.get_f32(f"{pfx}.self_attn.q_norm.weight")),
            ffn_dim=layer_ffn,
            gateup_proj=_q8b(b, np.concatenate([gate_w, up_w], axis=0)),
            down_proj=_q8b(b, loader.get_f32(f"{pfx}.mlp.down_proj.weight")),
            pli_gate=_q8b(b, loader.get_f32(f"{pfx}.per_layer_input_gate.weight")),
            pli_proj=_q8b(b, loader.get_f32(f"{pfx}.per_layer_projection.weight")),
        )
        if is_source:
            lw.qkv_proj = _q8b(b, np.concatenate([
                loader.get_f32(f"{pfx}.self_attn.q_proj.weight"),
                loader.get_f32(f"{pfx}.self_attn.k_proj.weight"),
                loader.get_f32(f"{pfx}.self_attn.v_proj.weight"),
            ], axis=0))
            lw.k_norm = _f32(b, loader.get_f32(f"{pfx}.self_attn.k_norm.weight"))
        else:
            lw.q_proj = _q8b(b, loader.get_f32(f"{pfx}.self_attn.q_proj.weight"))
        layers.append(lw)

    return shared, layers


# --------------------------------- forward -----------------------------------


def _emit_forward(b: Builder, shared: _SharedWeights, layers: List[_LayerWeights]):
    """Emit the full forward graph. Returns (logits, {src: (k_in, k_out, v_in, v_out)})."""
    c = _Consts(b)

    # Public runtime inputs.
    tokens = _input(b, "tokens", "i32", ("batch", "seq"))
    positions = _input(b, "positions", "i32", ("batch", "seq"))
    cache_write_index = _input(b, "cache_write_index", "i32", ("batch",))
    cache_visible_end = _input(b, "cache_visible_end", "i32", ("batch",))

    # KV cache public inputs — f16, one pair per source layer.
    k_cache_in: Dict[int, Value] = {}
    v_cache_in: Dict[int, Value] = {}
    for src in SOURCE_LAYERS:
        is_glob = is_global_layer(src)
        head_dim = GLOBAL_HEAD_DIM if is_glob else LOCAL_HEAD_DIM
        t_dim: Union[int, str] = "G" if is_glob else LOCAL_SLIDING_WINDOW
        k_cache_in[src] = _input(b, f"k_cache.layer{src}", "f16", ("batch", NUM_KV_HEADS, t_dim, head_dim))
        v_cache_in[src] = _input(b, f"v_cache.layer{src}", "f16", ("batch", NUM_KV_HEADS, t_dim, head_dim))

    # ---- Embedding + PLI encoder ----
    emb = b.gather_rows(shared.embed_tokens, tokens)                    # [B,S,1536] f32
    emb_scaled = _scalar_mul(b, c, emb, EMBED_DIM, math.sqrt(EMBED_DIM))

    pli_proj_flat = b.matmul(emb_scaled, shared.per_layer_model_projection)  # [B,S,8960]
    pli_proj_scaled = _scalar_mul(b, c, pli_proj_flat, PLI_TOTAL, 1.0 / math.sqrt(EMBED_DIM))
    pli_proj_r = b.reshape(pli_proj_scaled, ("batch", "seq", NUM_LAYERS, PLI_DIM))
    pli_proj_norm = _rmsnorm(b, c, pli_proj_r, shared.per_layer_projection_norm, PLI_DIM)

    pli_emb_flat = b.gather_rows(shared.embed_tokens_per_layer, tokens)  # [B,S,8960]
    pli_emb_scaled = _scalar_mul(b, c, pli_emb_flat, PLI_TOTAL, math.sqrt(PLI_DIM))
    pli_emb_r = b.reshape(pli_emb_scaled, ("batch", "seq", NUM_LAYERS, PLI_DIM))

    pli_sum = b.add(pli_proj_norm, pli_emb_r)
    pli = _scalar_mul(b, c, pli_sum, PLI_DIM, 1.0 / math.sqrt(2.0))      # [B,S,35,256]

    x = emb_scaled
    k_cache_cur: Dict[int, Value] = dict(k_cache_in)
    v_cache_cur: Dict[int, Value] = dict(v_cache_in)

    # ---- Per-layer transformer blocks ----
    for layer_idx, lw in enumerate(layers):
        is_glob = is_global_layer(layer_idx)
        head_dim = GLOBAL_HEAD_DIM if is_glob else LOCAL_HEAD_DIM
        sliding = 0 if is_glob else LOCAL_SLIDING_WINDOW
        rope_base = ROPE_GLOBAL_BASE if is_glob else ROPE_LOCAL_BASE
        rope_prop = ROPE_GLOBAL_PROPORTION if is_glob else ROPE_LOCAL_PROPORTION
        src = kv_source_of(layer_idx)
        is_source = layer_idx in SOURCE_LAYERS

        x_norm = _rmsnorm(b, c, x, lw.input_ln, EMBED_DIM)

        q_out = NUM_HEADS * head_dim
        kv_out = NUM_KV_HEADS * head_dim
        k_flat: Optional[Value] = None
        v_flat: Optional[Value] = None
        if is_source:
            assert lw.qkv_proj is not None
            qkv = b.matmul(x_norm, lw.qkv_proj)
            q_flat = b.slice(qkv, (0, 0, 0), ("batch", "seq", q_out))
            k_flat = b.slice(qkv, (0, 0, q_out), ("batch", "seq", kv_out))
            v_flat = b.slice(qkv, (0, 0, q_out + kv_out), ("batch", "seq", kv_out))
        else:
            assert lw.q_proj is not None
            q_flat = b.matmul(x_norm, lw.q_proj)

        q = b.reshape(q_flat, ("batch", "seq", NUM_HEADS, head_dim))
        q = _rmsnorm(b, c, q, lw.q_norm, head_dim)
        q = b.rope1d(q, positions, base_frequency=rope_base, rope_proportion=rope_prop)

        if is_source:
            assert lw.k_norm is not None and k_flat is not None and v_flat is not None
            k4 = b.reshape(k_flat, ("batch", "seq", NUM_KV_HEADS, head_dim))
            k4 = _rmsnorm(b, c, k4, lw.k_norm, head_dim)
            k4 = b.rope1d(k4, positions, base_frequency=rope_base, rope_proportion=rope_prop)
            k4_t = b.reshape(k4, ("batch", NUM_KV_HEADS, "seq", head_dim))
            k_f16 = b.cast(k4_t, "f16")
            k_cache_cur[layer_idx] = b.sequence_append(k_cache_cur[layer_idx], k_f16, cache_write_index)

            v4 = b.reshape(v_flat, ("batch", "seq", NUM_KV_HEADS, head_dim))
            # Parameterless V-norm (gamma implicitly all-ones).
            v4 = _rmsnorm(b, c, v4, c.scale_vec(head_dim, 1.0), head_dim)
            v4_t = b.reshape(v4, ("batch", NUM_KV_HEADS, "seq", head_dim))
            v_f16 = b.cast(v4_t, "f16")
            v_cache_cur[layer_idx] = b.sequence_append(v_cache_cur[layer_idx], v_f16, cache_write_index)

        attn_out = b.mha_cached(
            q, k_cache_cur[src], v_cache_cur[src], positions, cache_visible_end,
            scale=1.0, sliding_window=sliding, attn_logits_soft_cap=0.0,
        )

        attn_flat = b.reshape(attn_out, ("batch", "seq", NUM_HEADS * head_dim))
        o_out = b.matmul(attn_flat, lw.o_proj)
        o_out = _rmsnorm(b, c, o_out, lw.post_attn_ln, EMBED_DIM)
        x = b.add(x, o_out)

        ff_in = _rmsnorm(b, c, x, lw.pre_ffn_ln, EMBED_DIM)
        ffn = lw.ffn_dim
        assert lw.gateup_proj is not None
        gate_up = b.matmul(ff_in, lw.gateup_proj)
        gate = b.slice(gate_up, (0, 0, 0), ("batch", "seq", ffn))
        up = b.slice(gate_up, (0, 0, ffn), ("batch", "seq", ffn))
        gate_act = b.unary("gelu", gate)
        ff = b.mul(gate_act, up)
        ff = b.matmul(ff, lw.down_proj)
        ff = _rmsnorm(b, c, ff, lw.post_ffn_ln, EMBED_DIM)
        x = b.add(x, ff)

        pli_gate_in = b.matmul(x, lw.pli_gate)
        pli_gate_act = b.unary("gelu", pli_gate_in)
        pli_slice = b.slice(pli, (0, 0, layer_idx, 0), ("batch", "seq", 1, PLI_DIM))
        pli_slice2 = b.reshape(pli_slice, ("batch", "seq", PLI_DIM))
        pli_combined = b.mul(pli_gate_act, pli_slice2)
        pli_out = b.matmul(pli_combined, lw.pli_proj)
        pli_out = _rmsnorm(b, c, pli_out, lw.post_pli_ln, EMBED_DIM)
        x = b.add(x, pli_out)

        x = _scalar_mul(b, c, x, EMBED_DIM, lw.skip_scale)

    # ---- Tail ----
    x = _rmsnorm(b, c, x, shared.final_norm, EMBED_DIM)
    logits = b.matmul_nt(x, shared.embed_tokens)        # tied-embedding logits
    logits = _scalar_div(b, c, logits, VOCAB_SIZE, FINAL_LOGIT_SOFTCAP)
    logits = b.unary("tanh", logits)
    logits = _scalar_mul(b, c, logits, VOCAB_SIZE, FINAL_LOGIT_SOFTCAP)

    caches = {src: (k_cache_in[src], k_cache_cur[src], v_cache_in[src], v_cache_cur[src])
              for src in SOURCE_LAYERS}
    return logits, tokens, positions, cache_write_index, cache_visible_end, k_cache_in, v_cache_in, caches


def _finalize(b: Builder, logits: Value, tokens: Value, positions: Value,
              cache_write_index: Value, cache_visible_end: Value,
              k_cache_in: Dict[int, Value], v_cache_in: Dict[int, Value], caches) -> None:
    b.mark_output(logits, "logits")

    # next_*_cache outputs io-aliased back to the cache inputs.
    for src, (k_in, k_out, v_in, v_out) in caches.items():
        b.mark_output(k_out, f"next_k_cache.layer{src}")
        b.mark_output(v_out, f"next_v_cache.layer{src}")
        b.add_output_alias(k_in, k_out)
        b.add_output_alias(v_in, v_out)

    # Input roles: auto-fed indices/positions + auto-managed KV caches.
    b.add_input_role(tokens, RK.AION_ROLE_TOKENS, axis=1)
    b.add_input_role(positions, RK.AION_ROLE_POSITIONS, axis=1)
    b.add_input_role(cache_write_index, RK.AION_ROLE_CACHE_WRITE_INDEX)
    b.add_input_role(cache_visible_end, RK.AION_ROLE_CACHE_VISIBLE_END)
    for src in SOURCE_LAYERS:
        is_glob = is_global_layer(src)
        for v in (k_cache_in[src], v_cache_in[src]):
            if is_glob:
                # Global caches: free capacity symbol `G`, growable.
                b.add_input_role(v, RK.AION_ROLE_SEQUENCE_CACHE, axis=2,
                                 capacity_symbol="G", zero_init=True, growable=True)
            else:
                # Local caches: fixed sliding-window size.
                b.add_input_role(v, RK.AION_ROLE_SEQUENCE_CACHE, axis=2, zero_init=True)

    for key, value in [
        ("arch", "gemma4-e2b-text"),
        ("quant", "q8_0-weights-f16-cache"),
        ("num_layers", str(NUM_LAYERS)),
        ("embed_dim", str(EMBED_DIM)),
        ("local_head_dim", str(LOCAL_HEAD_DIM)),
        ("global_head_dim", str(GLOBAL_HEAD_DIM)),
        ("local_sliding_window", str(LOCAL_SLIDING_WINDOW)),
        ("final_logit_softcap", str(FINAL_LOGIT_SOFTCAP)),
        ("kv_sharing.local_source_layer", str(LOCAL_SOURCE_LAYER)),
        ("kv_sharing.global_source_layer", str(GLOBAL_SOURCE_LAYER)),
        ("graph_phase", "2-full-forward"),
    ]:
        b.add_metadata(key, value)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    with aion.Context(thread_count=1) as ctx:
        with Builder(ctx) as b:
            with _WeightLoader(args.in_safetensors) as loader:
                _reject_if_unexpected_multimodal(loader.keys(), args.allow_multimodal)
                shared, layers = _emit_weights(loader, b)
            out = _emit_forward(b, shared, layers)
            _finalize(b, *out)
            b.export(os.path.abspath(args.out_aion), None)

    print(f"[gemma4-e2b] wrote {args.out_aion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
