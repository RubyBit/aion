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
"""Convert Gemma 4 E2B safetensors -> one multimodal q8_0 `.aion`.

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
import gc
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import aion
from aion import Builder, TensorRef, nn
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
# Candidates the model hands back per step for host-side sampling. The top-k kernel
# costs one selection round per k, so this is a real per-token price: 32 measured
# ~1.5 tok/s faster than 64 on a 4080 and still covers any practical top-p nucleus.
TOPK_OUT: int = 32
FINAL_LOGIT_SOFTCAP: float = 30.0
PLI_DIM: int = 256
PLI_TOTAL: int = NUM_LAYERS * PLI_DIM  # 8960
FFN_DIM: int = 6144
RMS_EPS: float = 1e-6
ROPE_LOCAL_BASE: float = 10_000.0
ROPE_GLOBAL_BASE: float = 1_000_000.0
ROPE_LOCAL_PROPORTION: float = 1.0
ROPE_GLOBAL_PROPORTION: float = 0.25
# The reference masks `0 <= q - k < attention_context_left - 1`: 12 keys, not 13.
# The 13-entry relative table therefore has one row the mask never admits.
AUDIO_WINDOW = aion.AttentionWindow.sliding(11)
# Each head's first half rotates by the patch's x position, the second by its y.
VISION_ROPE = nn.AxialRoPE(base_frequency=100.0)


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
_PLACEHOLDER = {
    "batch": 1, "seq": 1,
    "G": LOCAL_SLIDING_WINDOW, "L": LOCAL_SLIDING_WINDOW,
    "vision_seq": 9, "vision_out": 1,
    "audio_frames": 4, "audio_sub1": 2, "audio_tokens": 1,
}

Shape = Tuple[Union[int, str], ...]


# ------------------------------- CLI + loading -------------------------------


def _parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert Gemma 4 E2B text/image/audio to one q8_0 .aion")
    ap.add_argument("in_safetensors", type=str)
    ap.add_argument("out_aion", type=str)
    return ap.parse_args(argv)


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


def _mm_b(w_torch: np.ndarray) -> np.ndarray:
    """A PyTorch linear's `[out, in]` weight in Aion's matmul-B layout `[in, out]`.

    Adapting someone else's layout is a converter's job — `nn` has one canonical
    layout and takes no `transposed=` flag. Rank 2 is all it needs: MatMul broadcasts
    a `[K, N]` weight into a `[batch, seq, K]` activation, so the weight reaches the
    kernel exactly as stored, which is what keeps a quantized one usable at all.
    Quantization itself is `nn`'s to do (`dtype=aion.q8_0`), blocking along K.
    """
    return np.ascontiguousarray(w_torch.astype(np.float32, copy=False).T)


def _f32(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def _input(
    b: Builder, name: str, dtype: aion.AionDType, shape: Shape
) -> TensorRef:
    """Declare a public input; string dims are symbolic (dynamic) axes."""
    dims = [_PLACEHOLDER[d] if isinstance(d, str) else int(d) for d in shape]
    dyn = {i: d for i, d in enumerate(shape) if isinstance(d, str)} or None
    return b.input(dims, dtype=dtype, dynamic=dyn).rename(name)


def _scaled(b: Builder, x: TensorRef, factor: float) -> TensorRef:
    """`x * factor`.

    The scalar is a *one-element* parameter the Builder caches and shares across the
    whole model, not a full-width vector: Aion has no scalar-typed operand, but the
    broadcast elementwise op takes a size-1 one. Before that, scaling a `[1, S, 1536]`
    activation meant baking a dense vector per distinct width and value.
    """
    return b.mul(x, b.constant(factor))


# Stage dumps for scripts/verify_gemma4_e2b.py: off unless the env var is set, so
# a normal conversion writes exactly the public outputs.
_DEBUG_STAGES: bool = os.environ.get("AION_GEMMA_DEBUG_STAGES") == "1"


def _dbg(b: Builder, value: TensorRef, name: str) -> TensorRef:
    """Mark `value` as a `dbg.<name>` output when stage dumping is enabled."""
    if _DEBUG_STAGES:
        b.mark_output(value, f"dbg.{name}")
    return value


def _rms(gamma: Optional[np.ndarray], name: str, width: Optional[int] = None) -> nn.RMSNorm:
    """A norm with Gemma's epsilon.

    An omitted gamma is the parameterless V-norm: `nn` then supplies the shared
    all-ones identity, so the caller never bakes one, and only needs to say how wide
    the normalized axis is.
    """
    if gamma is not None:
        return nn.RMSNorm(gamma, eps=RMS_EPS, name=name)
    if width is None:
        raise ValueError(f"norm {name!r} has no gamma, so it needs an explicit width")
    return nn.RMSNorm(None, eps=RMS_EPS, name=name, normalized_shape=(width,))


# ------------------------------- weight structs ------------------------------


# Weights are held as *data*: `nn` layers bind and quantize them themselves, so
# nothing here needs a Builder. Q/K/V and gate/up stay separate, the way the
# checkpoint ships them — pre-concatenating them is a *fusion*, and the compiler
# does that (`opt/fuse_horizontal_matmul` rewrites matmuls sharing an operand into
# one wide matmul plus slices, numerically identically) and then frees the sources.


@dataclass
class _LayerWeights:
    input_ln: np.ndarray
    post_attn_ln: np.ndarray
    pre_ffn_ln: np.ndarray
    post_ffn_ln: np.ndarray
    post_pli_ln: np.ndarray
    skip_scale: float
    q_proj: np.ndarray
    o_proj: np.ndarray
    q_norm: np.ndarray
    gate_proj: np.ndarray
    up_proj: np.ndarray
    down_proj: np.ndarray
    pli_gate: np.ndarray
    pli_proj: np.ndarray
    # Only a layer that owns a KV cache projects K/V; the rest read the cache a
    # source layer produced. All-or-nothing, and `nn` reads which case this is off
    # the weights it is handed.
    k_proj: Optional[np.ndarray] = None
    v_proj: Optional[np.ndarray] = None
    k_norm: Optional[np.ndarray] = None


@dataclass
class _SharedWeights:
    embed_tokens: np.ndarray
    embed_tokens_per_layer: np.ndarray
    per_layer_model_projection: np.ndarray
    per_layer_projection_norm: np.ndarray
    final_norm: np.ndarray


@dataclass
class _TowerWeights:
    values: Dict[str, np.ndarray]

    def get(self, name: str) -> np.ndarray:
        try:
            return self.values[name]
        except KeyError as exc:
            raise KeyError(f"multimodal checkpoint tensor is missing: {name}") from exc


def _load_weights(loader: _WeightLoader) -> Tuple[_SharedWeights, List[_LayerWeights]]:
    ln = "model.language_model"

    shared = _SharedWeights(
        embed_tokens=_f32(loader.get_f32(f"{ln}.embed_tokens.weight")),
        embed_tokens_per_layer=_f32(loader.get_f32(f"{ln}.embed_tokens_per_layer.weight")),
        per_layer_model_projection=_mm_b(loader.get_f32(f"{ln}.per_layer_model_projection.weight")),
        per_layer_projection_norm=_f32(loader.get_f32(f"{ln}.per_layer_projection_norm.weight")),
        final_norm=_f32(loader.get_f32(f"{ln}.norm.weight")),
    )

    layers: List[_LayerWeights] = []
    for layer in range(NUM_LAYERS):
        pfx = f"{ln}.layers.{layer}"

        lw = _LayerWeights(
            input_ln=_f32(loader.get_f32(f"{pfx}.input_layernorm.weight")),
            post_attn_ln=_f32(loader.get_f32(f"{pfx}.post_attention_layernorm.weight")),
            pre_ffn_ln=_f32(loader.get_f32(f"{pfx}.pre_feedforward_layernorm.weight")),
            post_ffn_ln=_f32(loader.get_f32(f"{pfx}.post_feedforward_layernorm.weight")),
            post_pli_ln=_f32(loader.get_f32(f"{pfx}.post_per_layer_input_norm.weight")),
            skip_scale=float(loader.get_f32(f"{pfx}.layer_scalar").reshape(-1)[0]),
            q_proj=_mm_b(loader.get_f32(f"{pfx}.self_attn.q_proj.weight")),
            o_proj=_mm_b(loader.get_f32(f"{pfx}.self_attn.o_proj.weight")),
            q_norm=_f32(loader.get_f32(f"{pfx}.self_attn.q_norm.weight")),
            # The FFN width is elastic per layer, and comes off the weight itself.
            gate_proj=_mm_b(loader.get_f32(f"{pfx}.mlp.gate_proj.weight")),
            up_proj=_mm_b(loader.get_f32(f"{pfx}.mlp.up_proj.weight")),
            down_proj=_mm_b(loader.get_f32(f"{pfx}.mlp.down_proj.weight")),
            pli_gate=_mm_b(loader.get_f32(f"{pfx}.per_layer_input_gate.weight")),
            pli_proj=_mm_b(loader.get_f32(f"{pfx}.per_layer_projection.weight")),
        )
        if layer in SOURCE_LAYERS:
            lw.k_proj = _mm_b(loader.get_f32(f"{pfx}.self_attn.k_proj.weight"))
            lw.v_proj = _mm_b(loader.get_f32(f"{pfx}.self_attn.v_proj.weight"))
            lw.k_norm = _f32(loader.get_f32(f"{pfx}.self_attn.k_norm.weight"))
        layers.append(lw)

    return shared, layers


def _load_tower_weights(loader: _WeightLoader) -> _TowerWeights:
    prefixes = ("model.vision_tower.", "model.audio_tower.", "model.embed_vision.", "model.embed_audio.")
    keys = [key for key in loader.keys() if key.startswith(prefixes) and key.endswith((".weight", ".bias", "position_embedding_table", "per_dim_scale"))]
    if not any(key.startswith("model.vision_tower.") for key in keys):
        raise SystemExit("checkpoint has no Gemma 4 vision tower")
    if not any(key.startswith("model.audio_tower.") for key in keys):
        raise SystemExit("checkpoint has no Gemma 4 audio tower")
    return _TowerWeights({key: _f32(loader.get_f32(key)) for key in keys})


def _linear_weight(towers: _TowerWeights, prefix: str) -> np.ndarray:
    """Read either a plain Linear or Gemma4ClippableLinear checkpoint weight."""
    direct = f"{prefix}.weight"
    wrapped = f"{prefix}.linear.weight"
    if direct in towers.values:
        return _mm_b(towers.get(direct))
    return _mm_b(towers.get(wrapped))


def _emit_vision_tower(
    b: Builder, towers: _TowerWeights, pixel_values: TensorRef,
    pixel_pos: Tuple[TensorRef, TensorRef], vision_lengths: TensorRef,
    pool_map: TensorRef, embed_map: TensorRef,
) -> TensorRef:
    pixel_x, pixel_y = pixel_pos
    vp = "model.vision_tower"
    x = _scaled(b, pixel_values, 2.0) - b.constant(1.0)
    x = nn.Linear(_linear_weight(towers, f"{vp}.patch_embedder.input_proj"),
                  dtype=aion.q8_0, name="patch_embedder.input_proj")(x)
    pos = towers.get(f"{vp}.patch_embedder.position_embedding_table")
    x_pos = nn.Embedding(pos[0], dtype=aion.q8_0, name="patch_embedder.x_pos")(pixel_x)
    y_pos = nn.Embedding(pos[1], dtype=aion.q8_0, name="patch_embedder.y_pos")(pixel_y)
    x = x + x_pos + y_pos
    _dbg(b, x, "vis.patch")

    for layer in range(16):
        p = f"{vp}.encoder.layers.{layer}"
        with b.scope(f"vision.layers.{layer}"):
            residual = x
            h = _rms(towers.get(f"{p}.input_layernorm.weight"), "input_layernorm")(x)
            attn = nn.Attention(
                _linear_weight(towers, f"{p}.self_attn.q_proj"),
                _linear_weight(towers, f"{p}.self_attn.o_proj"),
                k_proj=_linear_weight(towers, f"{p}.self_attn.k_proj"),
                v_proj=_linear_weight(towers, f"{p}.self_attn.v_proj"),
                heads=12, kv_heads=12, head_dim=64, scale=1.0,
                window=aion.AttentionWindow.FULL,
                dtype=aion.q8_0, name="self_attn",
            )
            q, k, v = attn.project(h)
            assert k is not None and v is not None
            q = VISION_ROPE.forward(_rms(towers.get(f"{p}.self_attn.q_norm.weight"), "q_norm")(q), pixel_pos)
            k = VISION_ROPE.forward(_rms(towers.get(f"{p}.self_attn.k_norm.weight"), "k_norm")(k), pixel_pos)
            v = _rms(None, "v_norm", 64)(v)
            h = attn.attend(q, k, v, kv_lengths=vision_lengths)
            x = residual + _rms(towers.get(f"{p}.post_attention_layernorm.weight"), "post_attention_layernorm")(h)

            residual = x
            h = _rms(towers.get(f"{p}.pre_feedforward_layernorm.weight"), "pre_feedforward_layernorm")(x)
            h = nn.GatedMLP(
                _linear_weight(towers, f"{p}.mlp.gate_proj"),
                _linear_weight(towers, f"{p}.mlp.up_proj"),
                _linear_weight(towers, f"{p}.mlp.down_proj"),
                act="gelu", dtype=aion.q8_0, name="mlp",
            )(h)
            x = residual + _rms(towers.get(f"{p}.post_feedforward_layernorm.weight"), "post_feedforward_layernorm")(h)
            if layer == 0:
                _dbg(b, x, "vis.layer0")
    _dbg(b, x, "vis.last")

    # Host-provided geometry matrix is the exact sparse k×k spatial average.
    x = _scaled(b, b.matmul(pool_map, x), math.sqrt(768.0))
    _dbg(b, x, "vis.pooled")
    x = _rms(None, "embed_vision.pre_projection_norm", 768)(x)
    x = nn.Linear(_linear_weight(towers, "model.embed_vision.embedding_projection"),
                  dtype=aion.q8_0, name="embed_vision.projection")(x)
    _dbg(b, x, "vis.features")
    return b.matmul(embed_map, x)


AUDIO_K_SCALE: float = math.log1p(math.e) / math.log(2.0)


def _audio_q_weight(towers: _TowerWeights, prefix: str) -> np.ndarray:
    """q_proj with Gemma's per-dim query scale folded in, so attention takes one scalar."""
    per_dim = towers.get(f"{prefix}.self_attn.per_dim_scale")
    scale = (128.0 ** -0.5) / math.log(2.0) * np.logaddexp(0.0, per_dim)
    return _linear_weight(towers, f"{prefix}.self_attn.q_proj") * np.tile(scale, 8)[None, :]


def _audio_position_table(towers: _TowerWeights, layer: int) -> np.ndarray:
    inv = np.exp(np.arange(512, dtype=np.float32) * (-math.log(10000.0) / 511.0))
    positions = np.arange(12, -1, -1, dtype=np.float32)[:, None]
    base = np.concatenate((np.sin(positions * inv), np.cos(positions * inv)), axis=-1)
    w = towers.get(f"model.audio_tower.layers.{layer}.self_attn.relative_k_proj.weight")
    return _f32((base @ w.T).reshape(13, 8, 128).transpose(1, 0, 2))


def _audio_ffn(b: Builder, towers: _TowerWeights, x: TensorRef, prefix: str) -> TensorRef:
    residual = x
    x = _rms(towers.get(f"{prefix}.pre_layer_norm.weight"), "pre_layer_norm")(x)
    x = nn.Linear(_linear_weight(towers, f"{prefix}.ffw_layer_1"), dtype=aion.q8_0, name="linear_1")(x).silu()
    x = nn.Linear(_linear_weight(towers, f"{prefix}.ffw_layer_2"), dtype=aion.q8_0, name="linear_2")(x)
    x = _scaled(b, _rms(towers.get(f"{prefix}.post_layer_norm.weight"), "post_layer_norm")(x), 0.5)
    return residual + x


def _emit_audio_tower(
    b: Builder, towers: _TowerWeights, features: TensorRef, feature_mask: TensorRef,
    subsample1_mask: TensorRef, attention_mask: TensorRef, embed_map: TensorRef,
) -> TensorRef:
    ap = "model.audio_tower"
    x = (features * feature_mask).reshape((1, "audio_frames", 128, 1))
    for idx, cin, cout in ((0, 1, 128), (1, 128, 32)):
        p = f"{ap}.subsample_conv_projection.layer{idx}"
        if idx == 1:
            x = x * subsample1_mask
        w = towers.get(f"{p}.conv.weight").transpose(2, 3, 1, 0)
        x = nn.Conv2D(_f32(w), stride_h=2, stride_w=2,
                      pad_top=1, pad_bottom=1, pad_left=1, pad_right=1,
                      name=f"audio.subsample.{idx}.conv")(x)
        x = nn.LayerNorm(towers.get(f"{p}.norm.weight"), eps=RMS_EPS,
                         name=f"audio.subsample.{idx}.norm")(x).relu()
    x = x.reshape((1, "audio_tokens", 1024))
    x = nn.Linear(_linear_weight(towers, f"{ap}.subsample_conv_projection.input_proj_linear"),
                  dtype=aion.q8_0, name="audio.input_projection")(x)
    _dbg(b, x, "aud.subsample")

    zero_bias = np.zeros((8, 128), dtype=np.float32)
    for layer in range(12):
        p = f"{ap}.layers.{layer}"
        with b.scope(f"audio.layers.{layer}"):
            x = _audio_ffn(b, towers, x, f"{p}.feed_forward1")
            if layer == 0:
                _dbg(b, x, "aud.l0.ffn1")
            residual = x
            h = nn.RelPosSelfAttention(
                _audio_q_weight(towers, p),
                _linear_weight(towers, f"{p}.self_attn.k_proj") * AUDIO_K_SCALE,
                _linear_weight(towers, f"{p}.self_attn.v_proj"),
                _linear_weight(towers, f"{p}.self_attn.post"),
                _audio_position_table(towers, layer), zero_bias, zero_bias,
                scale=1.0, mask=attention_mask, window=AUDIO_WINDOW,
                relative_zero_index=12, attn_logits_soft_cap=50.0,
                dtype=aion.q8_0, name="self_attn",
            )(_rms(towers.get(f"{p}.norm_pre_attn.weight"), "norm_pre_attn")(x))
            if layer == 0:
                _dbg(b, h, "aud.l0.attn")
            x = residual + _rms(towers.get(f"{p}.norm_post_attn.weight"), "norm_post_attn")(h)
            if layer == 0:
                _dbg(b, x, "aud.l0.post_attn")

            residual = x
            h = _rms(towers.get(f"{p}.lconv1d.pre_layer_norm.weight"), "lconv1d.pre_norm")(x)
            h = nn.Linear(_linear_weight(towers, f"{p}.lconv1d.linear_start"), dtype=aion.q8_0, name="lconv1d.linear_start")(h)
            # `F.glu(x)` is `first * sigmoid(second)` -- the value is the FIRST half
            # and the gate the second, which is the opposite of the reading that
            # comes naturally from "gated".
            h = b.slice_last_dim(h, 0, 1024) * b.slice_last_dim(h, 1024, 1024).sigmoid()
            cw = towers.get(f"{p}.lconv1d.depthwise_conv1d.weight").transpose(2, 1, 0)
            h = nn.Conv1D(_f32(cw), pad_left=4, groups=1024, name="lconv1d.depthwise")(h)
            h = _rms(towers.get(f"{p}.lconv1d.conv_norm.weight"), "lconv1d.conv_norm")(h).silu()
            h = nn.Linear(_linear_weight(towers, f"{p}.lconv1d.linear_end"), dtype=aion.q8_0, name="lconv1d.linear_end")(h)
            x = residual + h
            if layer == 0:
                _dbg(b, x, "aud.l0.lconv")
            x = _audio_ffn(b, towers, x, f"{p}.feed_forward2")
            if layer == 0:
                _dbg(b, x, "aud.l0.ffn2")
            x = _rms(towers.get(f"{p}.norm_out.weight"), "norm_out")(x)
            if layer == 0:
                _dbg(b, x, "aud.layer0")

    x = nn.Linear(_linear_weight(towers, f"{ap}.output_proj"), towers.get(f"{ap}.output_proj.bias"),
                  dtype=aion.q8_0, name="audio.output_proj")(x)
    _dbg(b, x, "aud.tower")
    x = _rms(None, "embed_audio.pre_projection_norm", 1536)(x)
    x = nn.Linear(_linear_weight(towers, "model.embed_audio.embedding_projection"),
                  dtype=aion.q8_0, name="embed_audio.projection")(x)
    _dbg(b, x, "aud.features")
    return b.matmul(embed_map, x)


# --------------------------------- forward -----------------------------------


def _last_index(b: Builder, tokens: TensorRef, x: TensorRef) -> TensorRef:
    """`[batch, 1]` i32 holding `seq - 1`, the row generation samples from.

    `b.dim` reifies the runtime sequence length as a one-element tensor. The `1`
    to subtract comes from `col <= col`, which is a batch-shaped i32 one -- so the
    index broadcasts to every batch entry (no `batch == 1` assumption) and the
    model carries no baked integer weight for it.
    """
    col = b.slice(tokens, (0, 0), ("batch", 1))
    return b.sub(b.dim(x, 1), b.compare("le", col, col))


def _emit_forward(b: Builder, shared: _SharedWeights, layers: List[_LayerWeights], towers: _TowerWeights):
    """Emit the full forward graph. Returns (logits, {src: (k_in, k_out, v_in, v_out)})."""
    # Public runtime inputs.
    tokens = _input(b, "tokens", aion.int32, ("batch", "seq"))
    positions = _input(b, "positions", aion.int32, ("batch", "seq"))
    cache_write_index = _input(
        b, "cache_write_index", aion.int32, ("batch",)
    )
    cache_visible_end = _input(
        b, "cache_visible_end", aion.int32, ("batch",)
    )

    # Processor outputs and sparse placement matrices. Towers live behind regions,
    # so text-only decode never evaluates them; dummy tensors only satisfy binding.
    has_image = _input(b, "has_image", aion.int32, (1,))
    text_embedding_mask = _input(b, "text_embedding_mask", aion.float32, ("batch", "seq", 1))
    pixel_values = _input(b, "pixel_values", aion.float32, (1, "vision_seq", 768))
    pixel_x = _input(b, "image_position_x", aion.int32, (1, "vision_seq"))
    pixel_y = _input(b, "image_position_y", aion.int32, (1, "vision_seq"))
    vision_lengths = _input(b, "vision_lengths", aion.int32, (1,))
    image_pool_map = _input(b, "image_pool_map", aion.float32, (1, "vision_out", "vision_seq"))
    image_embed_map = _input(b, "image_embed_map", aion.float32, ("batch", "seq", "vision_out"))

    has_audio = _input(b, "has_audio", aion.int32, (1,))
    audio_features = _input(b, "input_features", aion.float32, (1, "audio_frames", 128))
    audio_feature_mask = _input(b, "input_features_mask", aion.float32, (1, "audio_frames", 1))
    audio_subsample1_mask = _input(b, "audio_subsample1_mask", aion.float32, (1, "audio_sub1", 1, 1))
    audio_attention_mask = _input(b, "audio_attention_mask", aion.float32, ("audio_tokens", "audio_tokens"))
    audio_embed_map = _input(b, "audio_embed_map", aion.float32, ("batch", "seq", "audio_tokens"))

    # KV cache public inputs — f16, one pair per source layer.
    k_cache_in: Dict[int, TensorRef] = {}
    v_cache_in: Dict[int, TensorRef] = {}
    for src in SOURCE_LAYERS:
        is_glob = is_global_layer(src)
        head_dim = GLOBAL_HEAD_DIM if is_glob else LOCAL_HEAD_DIM
        # Both cache kinds have a runtime-sized physical capacity, on separate
        # symbols: `G` follows the whole context, `L` only ever holds the retained
        # 511 tokens plus room for the widest append seen.
        t_dim: Union[int, str] = "G" if is_glob else "L"
        k_cache_in[src] = _input(
            b,
            f"k_cache.layer{src}",
            aion.float16,
            ("batch", t_dim, NUM_KV_HEADS, head_dim),
        )
        v_cache_in[src] = _input(
            b,
            f"v_cache.layer{src}",
            aion.float16,
            ("batch", t_dim, NUM_KV_HEADS, head_dim),
        )

    # ---- Embedding + PLI encoder ----
    # The token table is tied to the output head, so it is bound once here and the
    # head reuses that very parameter rather than a second copy of it.
    embed = nn.Embedding(shared.embed_tokens, dtype=aion.q8_0, name="embed_tokens")
    emb_scaled = _scaled(b, embed(tokens), math.sqrt(EMBED_DIM))         # [B,S,1536]

    _dbg(b, emb_scaled, "emb_scaled")

    if _DEBUG_STAGES:
        # Stage dumping needs the tower stages to be ordinary graph values, which a
        # region output is not. The towers then always execute; only the validation
        # build takes this path.
        vision_sparse = _emit_vision_tower(
            b, towers, pixel_values, (pixel_x, pixel_y), vision_lengths,
            image_pool_map, image_embed_map,
        )
        audio_sparse = _emit_audio_tower(
            b, towers, audio_features, audio_feature_mask, audio_subsample1_mask,
            audio_attention_mask, audio_embed_map,
        )
    else:
        b.begin_region()
        vision_sparse = _emit_vision_tower(
            b, towers, pixel_values, (pixel_x, pixel_y), vision_lengths,
            image_pool_map, image_embed_map,
        )
        vision_region = b.end_region((vision_sparse,))
        b.begin_region()
        no_vision_region = b.end_region((_scaled(b, emb_scaled, 0.0),))

        b.begin_region()
        audio_sparse = _emit_audio_tower(
            b, towers, audio_features, audio_feature_mask, audio_subsample1_mask,
            audio_attention_mask, audio_embed_map,
        )
        audio_region = b.end_region((audio_sparse,))
        b.begin_region()
        no_audio_region = b.end_region((_scaled(b, emb_scaled, 0.0),))

        vision_sparse = b.if_(has_image, vision_region, no_vision_region)
        audio_sparse = b.if_(has_audio, audio_region, no_audio_region)
    # Placeholder ids are replaced with PAD for safe gathers and PLI construction;
    # this mask removes PAD's main embedding where a soft token is inserted.
    x = emb_scaled * text_embedding_mask + vision_sparse + audio_sparse
    _dbg(b, x, "merged")

    # ---- PLI ----
    # Two halves, and they read different things. The token-identity half comes
    # from the ids (placeholders already replaced with PAD upstream); the
    # context half projects the embedding the decoder actually consumes -- the
    # MERGED one, soft tokens included, which is what the reference feeds its
    # text model. Building it before the merge would leave every image/audio slot
    # projecting PAD.
    pli_proj = nn.Linear(shared.per_layer_model_projection, dtype=aion.q8_0,
                         name="per_layer_model_projection")
    pli_proj_scaled = _scaled(b, pli_proj(x), 1.0 / math.sqrt(EMBED_DIM))
    pli_proj_norm = _rms(shared.per_layer_projection_norm, "per_layer_projection_norm")(
        pli_proj_scaled.reshape(("batch", "seq", NUM_LAYERS, PLI_DIM)))

    pli_emb = nn.Embedding(shared.embed_tokens_per_layer, dtype=aion.q8_0,
                           name="embed_tokens_per_layer")
    pli_emb_r = _scaled(b, pli_emb(tokens), math.sqrt(PLI_DIM)).reshape(
        ("batch", "seq", NUM_LAYERS, PLI_DIM))

    pli = _scaled(b, pli_proj_norm + pli_emb_r, 1.0 / math.sqrt(2.0))     # [B,S,35,256]
    _dbg(b, pli, "pli")
    k_cache_cur: Dict[int, TensorRef] = dict(k_cache_in)
    v_cache_cur: Dict[int, TensorRef] = dict(v_cache_in)

    # ---- Per-layer transformer blocks ----
    for layer_idx, lw in enumerate(layers):
        is_glob = is_global_layer(layer_idx)
        head_dim = GLOBAL_HEAD_DIM if is_glob else LOCAL_HEAD_DIM
        window = (aion.AttentionWindow.CAUSAL if is_glob
                  else aion.AttentionWindow.sliding(LOCAL_SLIDING_WINDOW - 1))
        rope_base = ROPE_GLOBAL_BASE if is_glob else ROPE_LOCAL_BASE
        rope_prop = ROPE_GLOBAL_PROPORTION if is_glob else ROPE_LOCAL_PROPORTION
        src = kv_source_of(layer_idx)
        is_source = layer_idx in SOURCE_LAYERS

        # One scope per layer, so every parameter under it is named
        # `layers.{i}/<layer>/<param>` — the key a loaded model addresses it by.
        with b.scope(f"layers.{layer_idx}"):
            # A source layer projects K/V and writes its own cache; the layers after
            # it have no K/V projections and attend over the cache it produced.
            attn = nn.Attention(
                lw.q_proj, lw.o_proj,
                k_proj=lw.k_proj, v_proj=lw.v_proj,
                heads=NUM_HEADS, kv_heads=NUM_KV_HEADS, head_dim=head_dim,
                scale=1.0, window=window, attn_logits_soft_cap=0.0,
                dtype=aion.q8_0, name="self_attn",
            )

            x_norm = _rms(lw.input_ln, "input_layernorm")(x)
            if layer_idx == 0:
                _dbg(b, x_norm, "l0.input_ln")
            q, k4, v4 = attn.project(x_norm)
            if layer_idx == 0:
                _dbg(b, q, "l0.q_proj")
                if k4 is not None:
                    _dbg(b, k4, "l0.k_proj")
                if v4 is not None:
                    _dbg(b, v4, "l0.v_proj")

            # Q/K norms and RoPE sit between projection and attention — which is
            # exactly why `project` and `attend` are separate calls, and why all
            # three come back in this layout.
            q = _rms(lw.q_norm, "q_norm")(q)
            if layer_idx == 0:
                _dbg(b, q, "l0.q_norm")
            q = b.rope1d(q, positions, base_frequency=rope_base, rope_proportion=rope_prop)
            if layer_idx == 0:
                _dbg(b, q, "l0.q_rope")

            if is_source:
                assert k4 is not None and v4 is not None and lw.k_norm is not None
                k4 = _rms(lw.k_norm, "k_norm")(k4)
                if layer_idx == 0:
                    _dbg(b, k4, "l0.k_norm")
                k4 = b.rope1d(k4, positions, base_frequency=rope_base, rope_proportion=rope_prop)
                if layer_idx == 0:
                    _dbg(b, k4, "l0.k_rope")
                # A parameterless V-norm: with no gamma, `nn` supplies the shared
                # all-ones identity rather than making the caller bake one.
                v4 = _rms(None, "v_norm", head_dim)(v4)
                if layer_idx == 0:
                    _dbg(b, v4, "l0.v_norm")

                for kv, cur in ((k4, k_cache_cur), (v4, v_cache_cur)):
                    cur[layer_idx] = b.sequence_append(
                        cur[layer_idx], kv.cast(aion.float16), cache_write_index
                    )

            # `src` names this layer's cache owner — itself when it is a source.
            o_out = attn.attend(
                q, k_cache_cur[src], v_cache_cur[src],
                query_positions=positions, kv_lengths=cache_visible_end,
            )
            if layer_idx == 0:
                _dbg(b, o_out, "l0.o_proj")
            post_attn = _rms(lw.post_attn_ln, "post_attention_layernorm")(o_out)
            if layer_idx == 0:
                _dbg(b, post_attn, "l0.post_attn_ln")
            if layer_idx == 0:
                _dbg(b, x, "l0.residual_in")
            x = x + post_attn
            if layer_idx == 0:
                _dbg(b, x, "l0.x_after_attn")

            ff_in = _rms(lw.pre_ffn_ln, "pre_feedforward_layernorm")(x)
            if layer_idx == 0:
                _dbg(b, ff_in, "l0.pre_ffn_ln")
            # Split gate/up: the compiler fuses the matmul pair, and the GEGLU is one
            # `gate` op, so there is nothing to pre-concatenate or slice here.
            ff = nn.GatedMLP(lw.gate_proj, lw.up_proj, lw.down_proj,
                act="gelu", dtype=aion.q8_0, name="mlp")(ff_in)
            if layer_idx == 0:
                _dbg(b, ff, "l0.mlp")
            post_ffn = _rms(lw.post_ffn_ln, "post_feedforward_layernorm")(ff)
            if layer_idx == 0:
                _dbg(b, post_ffn, "l0.post_ffn_ln")
            x = x + post_ffn
            if layer_idx == 0:
                _dbg(b, x, "l0.x_after_ffn")

            pli_gate = nn.Linear(
                lw.pli_gate, dtype=aion.q8_0, name="per_layer_input_gate"
            )
            pli_out_proj = nn.Linear(
                lw.pli_proj, dtype=aion.q8_0, name="per_layer_projection"
            )
            pli_slice = b.slice(pli, (0, 0, layer_idx, 0), ("batch", "seq", 1, PLI_DIM))
            pli_gated = pli_gate(x)
            pli_out = pli_out_proj(
                pli_gated.gelu() * pli_slice.reshape(("batch", "seq", PLI_DIM)))
            post_pli = _rms(lw.post_pli_ln, "post_per_layer_input_norm")(pli_out)
            if layer_idx == 0:
                _dbg(b, pli_slice.reshape(("batch", "seq", PLI_DIM)), "l0.per_layer_input")
                _dbg(b, pli_gated, "l0.pli_gate")
                _dbg(b, pli_out, "l0.pli_proj")
                _dbg(b, post_pli, "l0.post_pli_ln")
            x = x + post_pli

            x = _scaled(b, x, lw.skip_scale)
            _dbg(b, x, f"layer{layer_idx}")

    # ---- Tail ----
    # Generation only ever needs the *last* position's distribution, and the tied
    # head is the single widest matmul in the model (vocab 262144). Keeping every
    # prefill row would make a 272-token prompt materialize a 285 MB f32 logits
    # tensor -- which argmax then has to see as whole rows, forcing a retile the
    # compiler rightly refuses. Select the final row first: decode (seq == 1)
    # gathers row 0 and is unchanged, prefill drops S-1 rows of head work.
    x = b.gather(x, _last_index(b, tokens, x), axis=1, batch_dims=1)

    x = _rms(shared.final_norm, "norm")(x)
    # Tied head: contract against the embedding table's *rows*, reusing the very
    # parameter `embed` bound rather than a second copy of the table.
    logits = b.matmul_nt(x, embed.weight_value(b))
    logits = b.div(logits, b.constant(FINAL_LOGIT_SOFTCAP)).tanh()
    logits = _scaled(b, logits, FINAL_LOGIT_SOFTCAP)

    caches = {src: (k_cache_in[src], k_cache_cur[src], v_cache_in[src], v_cache_cur[src])
              for src in SOURCE_LAYERS}
    return logits, tokens, positions, cache_write_index, cache_visible_end, k_cache_in, v_cache_in, caches


def _finalize(b: Builder, logits: TensorRef, tokens: TensorRef, positions: TensorRef,
              cache_write_index: TensorRef, cache_visible_end: TensorRef,
              k_cache_in: Dict[int, TensorRef], v_cache_in: Dict[int, TensorRef], caches) -> None:
    b.mark_output(logits, "logits")

    # Sampling needs a distribution, not one id -- but reading the [B, 1, 262144]
    # logits back per step is 1 MiB of transfer for the ~64 entries any sane
    # temperature/top-p leaves alive. Top-k on the device turns that into 512 bytes
    # and keeps the sampling POLICY (temperature, top-p, seed) on the host where it
    # belongs, rather than baking an RNG into the graph. Outputs are mirrored to the
    # host lazily, so leaving `logits` unread is what skips copying them.
    top_v, top_i = b.topk(logits, TOPK_OUT, axis=-1)
    b.mark_output(top_v, "topk_values")
    b.mark_output(top_i, "topk_indices")

    # Greedy next token. Top-k is sorted best-first, so this is its first column --
    # a view, not a second full pass over the vocabulary. An `argmax` here would
    # scan all 262144 logits again for a value top-k already computed.
    # `logits` is [batch, 1, vocab] -- the tail already gathered the sampled row --
    # so this is [batch, 1, 1] reshaped to the [batch, 1] the old argmax produced.
    b.mark_output(b.slice_last_dim(top_i, 0, 1).reshape(("batch", 1)), "next_token")

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
                b.add_input_role(v, RK.AION_ROLE_SEQUENCE_CACHE, axis=1,
                                 capacity_symbol="G", zero_init=True, growable=True)
            else:
                # Local caches: bounded semantic history, runtime-sized storage.
                b.add_input_role(v, RK.AION_ROLE_SEQUENCE_CACHE, axis=1,
                                 capacity_symbol="L", zero_init=True,
                                 retained_history_tokens=LOCAL_SLIDING_WINDOW - 1)

    for key, value in [
        ("arch", "gemma4-e2b-multimodal"),
        ("quant", "q8_0-weights-f16-cache"),
        ("num_layers", str(NUM_LAYERS)),
        ("embed_dim", str(EMBED_DIM)),
        ("local_head_dim", str(LOCAL_HEAD_DIM)),
        ("global_head_dim", str(GLOBAL_HEAD_DIM)),
        ("local_sliding_window", str(LOCAL_SLIDING_WINDOW)),
        ("final_logit_softcap", str(FINAL_LOGIT_SOFTCAP)),
        ("kv_sharing.local_source_layer", str(LOCAL_SOURCE_LAYER)),
        ("kv_sharing.global_source_layer", str(GLOBAL_SOURCE_LAYER)),
        ("graph_phase", "3-multimodal-forward"),
    ]:
        b.add_metadata(key, value)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    with aion.Context(thread_count=1) as ctx:
        with Builder(ctx) as b:
            with _WeightLoader(args.in_safetensors) as loader:
                shared, layers = _load_weights(loader)
                towers = _load_tower_weights(loader)
            out = _emit_forward(b, shared, layers, towers)
            _finalize(b, *out)
            # Binding/quantization copies weights into Aion-owned storage. Drop
            # the checkpoint-side f32 arrays before export, which itself packs
            # initializer sections and temporarily needs another large buffer.
            del shared, layers, towers
            gc.collect()
            b.export(os.path.abspath(args.out_aion), None)

    print(f"[gemma4-e2b] wrote {args.out_aion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
