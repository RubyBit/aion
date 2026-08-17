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


# --------------------------------- forward -----------------------------------


def _emit_forward(b: Builder, shared: _SharedWeights, layers: List[_LayerWeights]):
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

    # KV cache public inputs — f16, one pair per source layer.
    k_cache_in: Dict[int, TensorRef] = {}
    v_cache_in: Dict[int, TensorRef] = {}
    for src in SOURCE_LAYERS:
        is_glob = is_global_layer(src)
        head_dim = GLOBAL_HEAD_DIM if is_glob else LOCAL_HEAD_DIM
        t_dim: Union[int, str] = "G" if is_glob else LOCAL_SLIDING_WINDOW
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

    pli_proj = nn.Linear(shared.per_layer_model_projection, dtype=aion.q8_0,
                         name="per_layer_model_projection")
    pli_proj_scaled = _scaled(b, pli_proj(emb_scaled), 1.0 / math.sqrt(EMBED_DIM))
    pli_proj_norm = _rms(shared.per_layer_projection_norm, "per_layer_projection_norm")(
        pli_proj_scaled.reshape(("batch", "seq", NUM_LAYERS, PLI_DIM)))

    pli_emb = nn.Embedding(shared.embed_tokens_per_layer, dtype=aion.q8_0,
                           name="embed_tokens_per_layer")
    pli_emb_r = _scaled(b, pli_emb(tokens), math.sqrt(PLI_DIM)).reshape(
        ("batch", "seq", NUM_LAYERS, PLI_DIM))

    pli = _scaled(b, pli_proj_norm + pli_emb_r, 1.0 / math.sqrt(2.0))     # [B,S,35,256]

    x = emb_scaled
    k_cache_cur: Dict[int, TensorRef] = dict(k_cache_in)
    v_cache_cur: Dict[int, TensorRef] = dict(v_cache_in)

    # ---- Per-layer transformer blocks ----
    for layer_idx, lw in enumerate(layers):
        is_glob = is_global_layer(layer_idx)
        head_dim = GLOBAL_HEAD_DIM if is_glob else LOCAL_HEAD_DIM
        sliding = 0 if is_glob else LOCAL_SLIDING_WINDOW
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
                scale=1.0, sliding_window=sliding, attn_logits_soft_cap=0.0,
                dtype=aion.q8_0, name="self_attn",
            )

            x_norm = _rms(lw.input_ln, "input_layernorm")(x)
            q, k4, v4 = attn.project(x_norm)

            # Q/K norms and RoPE sit between projection and attention — which is
            # exactly why `project` and `attend` are separate calls, and why all
            # three come back in this layout.
            q = _rms(lw.q_norm, "q_norm")(q)
            q = b.rope1d(q, positions, base_frequency=rope_base, rope_proportion=rope_prop)

            if is_source:
                assert k4 is not None and v4 is not None and lw.k_norm is not None
                k4 = _rms(lw.k_norm, "k_norm")(k4)
                k4 = b.rope1d(k4, positions, base_frequency=rope_base, rope_proportion=rope_prop)
                # A parameterless V-norm: with no gamma, `nn` supplies the shared
                # all-ones identity rather than making the caller bake one.
                v4 = _rms(None, "v_norm", head_dim)(v4)

                for kv, cur in ((k4, k_cache_cur), (v4, v_cache_cur)):
                    cur[layer_idx] = b.sequence_append(
                        cur[layer_idx], kv.cast(aion.float16), cache_write_index
                    )

            # `src` names this layer's cache owner — itself when it is a source.
            o_out = attn.attend(
                q, k_cache_cur[src], v_cache_cur[src],
                query_positions=positions, kv_lengths=cache_visible_end,
            )
            x = x + _rms(lw.post_attn_ln, "post_attention_layernorm")(o_out)

            ff_in = _rms(lw.pre_ffn_ln, "pre_feedforward_layernorm")(x)
            # Split gate/up: the compiler fuses the pair and routes gelu through the
            # fused `gelu_mul`, so there is nothing to pre-concatenate or slice here.
            ff = nn.GatedMLP(lw.gate_proj, lw.up_proj, lw.down_proj,
                act="gelu", dtype=aion.q8_0, name="mlp")(ff_in)
            x = x + _rms(lw.post_ffn_ln, "post_feedforward_layernorm")(ff)

            pli_gate = nn.Linear(
                lw.pli_gate, dtype=aion.q8_0, name="per_layer_input_gate"
            )
            pli_out_proj = nn.Linear(
                lw.pli_proj, dtype=aion.q8_0, name="per_layer_projection"
            )
            pli_slice = b.slice(pli, (0, 0, layer_idx, 0), ("batch", "seq", 1, PLI_DIM))
            pli_out = pli_out_proj(
                pli_gate(x).gelu() * pli_slice.reshape(("batch", "seq", PLI_DIM)))
            x = x + _rms(lw.post_pli_ln, "post_per_layer_input_norm")(pli_out)

            x = _scaled(b, x, lw.skip_scale)

    # ---- Tail ----
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

    # Greedy next token, picked on the execution device. Outputs are mirrored to
    # the host lazily (only when read), so a caller that wants greedy decoding
    # reads 4 bytes here instead of copying the whole [B, T, vocab] logits back
    # every step; a caller that samples still has `logits`.
    b.mark_output(b.argmax(logits, axis=-1), "next_token")

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
                # Local caches: fixed sliding-window size.
                b.add_input_role(v, RK.AION_ROLE_SEQUENCE_CACHE, axis=1, zero_init=True)

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
                shared, layers = _load_weights(loader)
            out = _emit_forward(b, shared, layers)
            _finalize(b, *out)
            # Binding/quantization copies weights into Aion-owned storage. Drop
            # the checkpoint-side f32 arrays before export, which itself packs
            # initializer sections and temporarily needs another large buffer.
            del shared, layers
            gc.collect()
            b.export(os.path.abspath(args.out_aion), None)

    print(f"[gemma4-e2b] wrote {args.out_aion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
