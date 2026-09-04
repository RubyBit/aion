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
"""Convert `microsoft/harrier-oss-v1-270m` safetensors -> a self-contained q8_0 `.aion`.

Harrier-OSS v1 is a `Gemma3TextModel` (18 layers, d=640, GQA 4:1, head_dim=256)
repurposed as a text *embedding* model: last-token pooling + L2 normalization,
cosine similarity. So this emits ONE stateless encoder — token embedding + 18
causal transformer blocks + final norm + last-token pool + L2 normalize — and no
generative tail: no logits head, no KV cache, no io-aliased state.

That statelessness is the design: an encoder is a pure function of its tokens, so
nothing carries between runs and nothing has to be reset between two sentences.
Positions and per-row sequence lengths are derived in-graph, leaving only tokens
and an attention mask as public inputs.

Reference implementations this follows:
- `Gemma3TextModel` in HF transformers (`transformers.models.gemma3`), and
- `backends/candle/src/models/gemma3.rs` in huggingface/text-embeddings-inference,
  which is the same architecture wired for embeddings (last-token / mean pooling).

Config facts that shape the graph (all asserted against `config.json`, see
`_verify_config`) — every layer is `full_attention`, so unlike a stock Gemma 3
there is *no* local/global alternation: one rope base (`rope_theta`), no sliding
window, and neither logit softcap is set.

Weight/quant policy (mirrors convert_gemma4_e2b_to_aion.py):
- dense matmul weights -> q8_0 stored matmul-B `[K, N]` (blocked on K).
- the embedding table -> q8_0 blocked along the feature dim.
- RMSNorm gammas -> f32.
Pass `--dtype f32` for an unquantized reference to measure q8_0 drift against.

Run:  uv run --project bindings/python \
        python scripts/convert_harrier_oss_v1_to_aion.py <model_dir|model.safetensors> <out.aion>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import aion
from aion import Builder, TensorRef, nn
from aion.nn import functional as F


# ------------------------------ Model constants ------------------------------

NUM_LAYERS: int = 18
EMBED_DIM: int = 640
VOCAB_SIZE: int = 262144
NUM_HEADS: int = 4
NUM_KV_HEADS: int = 1
HEAD_DIM: int = 256
FFN_DIM: int = 2048
RMS_EPS: float = 1e-6
ROPE_BASE: float = 1_000_000.0
MAX_POSITIONS: int = 32768
# `1 / sqrt(query_pre_attn_scalar)`, which happens to equal `1 / sqrt(head_dim)`
# here — Gemma 3 keeps them separate config knobs, so derive it from the one the
# model actually scales by.
QUERY_PRE_ATTN_SCALAR: int = 256
ATTN_SCALE: float = 1.0 / math.sqrt(QUERY_PRE_ATTN_SCALAR)

# Special ids + tokenizer behavior, recorded in metadata so a driver configures
# itself from the package. `add_eos_token` matters for correctness here: the
# tokenizer appends `<eos>`, so under last-token pooling the pooled position IS
# the `<eos>` token, and dropping it changes the embedding.
BOS_TOKEN_ID: int = 2
EOS_TOKEN_ID: int = 1
PAD_TOKEN_ID: int = 0

# The instruction prefixes the model card ships. Queries need one; documents do
# not. Carried into metadata verbatim so the caller does not have to re-derive
# them from a config file that is not next to the `.aion`.
PROMPTS: Dict[str, str] = {
    "web_search_query": (
        "Instruct: Given a web search query, retrieve relevant passages that "
        "answer the query\nQuery: "
    ),
    "sts_query": "Instruct: Retrieve semantically similar text\nQuery: ",
    "bitext_query": "Instruct: Retrieve parallel sentences\nQuery: ",
}

# Authoring placeholders for symbolic axes (eager inference only); the exported
# model serves any positive batch and sequence length.
_PLACEHOLDER = {"batch": 2, "seq": 8}

# Batched pooling uses Gather(axis=1, batch_dims=1), so both axes stay symbolic.
# Inputs use right padding, and the mask supplies each row's valid length.

Shape = Tuple[Union[int, str], ...]


# ------------------------------- CLI + loading -------------------------------


def _parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Convert microsoft/harrier-oss-v1-270m to a q8_0 .aion embedding model"
    )
    ap.add_argument(
        "in_path",
        type=str,
        help="HF snapshot directory, or the model.safetensors inside one "
        "(config.json is read from alongside it when present)",
    )
    ap.add_argument("out_aion", type=str)
    ap.add_argument(
        "--dtype",
        choices=["q8_0", "f32"],
        default="q8_0",
        help="Weight dtype for matmuls and the embedding table (default: q8_0). "
        "f32 produces an unquantized reference ~4x larger.",
    )
    ap.add_argument(
        "--skip-config-check",
        action="store_true",
        help="Do not verify config.json against this converter's constants.",
    )
    return ap.parse_args(argv)


def _resolve_paths(in_path: str) -> Tuple[str, Optional[str]]:
    """`(safetensors_path, config_path_or_None)` from a directory or file path."""
    if os.path.isdir(in_path):
        st = os.path.join(in_path, "model.safetensors")
        if not os.path.isfile(st):
            raise SystemExit(f"{in_path}: no model.safetensors in this directory")
    else:
        st = in_path
        if not os.path.isfile(st):
            raise SystemExit(f"{st}: not a file")
    cfg = os.path.join(os.path.dirname(os.path.abspath(st)), "config.json")
    return st, cfg if os.path.isfile(cfg) else None


def _verify_config(cfg_path: str) -> None:
    """Fail loudly when the checkpoint's config contradicts this converter.

    Every constant above is baked into the emitted graph, so a config that
    disagrees means a silently wrong model — the failure mode a converter should
    never have. Checked here rather than trusted, because the interesting fields
    (`layer_types`, both softcaps, `use_bidirectional_attention`) are exactly the
    ones that would change the graph's *structure* rather than its shapes, and a
    shape mismatch is the only kind the Builder would catch on its own.
    """
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = json.load(fh)

    problems: List[str] = []

    def expect(key: str, want: Any) -> None:
        got = cfg.get(key, "<missing>")
        if got != want:
            problems.append(f"{key}: expected {want!r}, got {got!r}")

    expect("model_type", "gemma3_text")
    expect("num_hidden_layers", NUM_LAYERS)
    expect("hidden_size", EMBED_DIM)
    expect("vocab_size", VOCAB_SIZE)
    expect("num_attention_heads", NUM_HEADS)
    expect("num_key_value_heads", NUM_KV_HEADS)
    expect("head_dim", HEAD_DIM)
    expect("intermediate_size", FFN_DIM)
    expect("rms_norm_eps", RMS_EPS)
    expect("rope_theta", ROPE_BASE)
    expect("query_pre_attn_scalar", QUERY_PRE_ATTN_SCALAR)
    expect("max_position_embeddings", MAX_POSITIONS)
    expect("hidden_activation", "gelu_pytorch_tanh")
    expect("attention_bias", False)
    # Structural assumptions, in the order they would bite:
    #  - a sliding layer would need a second rope base and a windowed mask;
    #  - bidirectional attention would need a non-causal mask;
    #  - either softcap would need a tanh clamp the graph does not emit.
    expect("use_bidirectional_attention", False)
    expect("attn_logit_softcapping", None)
    expect("final_logit_softcapping", None)

    layer_types = cfg.get("layer_types")
    if layer_types != ["full_attention"] * NUM_LAYERS:
        distinct = sorted(set(layer_types)) if isinstance(layer_types, list) else layer_types
        problems.append(
            f"layer_types: expected all {NUM_LAYERS} layers 'full_attention', got {distinct!r} "
            "(a sliding-window layer needs rope_local_base_freq and a windowed mask)"
        )

    if problems:
        raise SystemExit(
            "config.json does not match this converter:\n  "
            + "\n  ".join(problems)
            + "\n\nRe-check the checkpoint, or pass --skip-config-check to convert anyway."
        )


class _WeightLoader:
    """safetensors reader that up-casts bf16 to f32 before handing values on."""

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


# --------------------- weight adaptation (host-side folds) --------------------


def _mm_b(w_torch: np.ndarray) -> np.ndarray:
    """A PyTorch linear's `[out, in]` weight in Aion's matmul-B layout `[in, out]`.

    Adapting someone else's layout is a converter's job — `nn` has one canonical
    layout and takes no `transposed=` flag. Rank 2 is all it needs: MatMul
    broadcasts a `[K, N]` weight into a `[batch, seq, K]` activation, so the
    weight reaches the kernel exactly as stored, which is what keeps a quantized
    one usable at all. Quantization itself is `nn`'s to do (`dtype=aion.q8_0`),
    blocking along K.
    """
    return np.ascontiguousarray(w_torch.astype(np.float32, copy=False).T)


def _gemma_gamma(w_torch: np.ndarray) -> np.ndarray:
    """A Gemma RMSNorm weight as a plain multiplicative gamma: `1 + w`.

    Every Gemma generation stores this weight zero-centered and scales by
    `(1.0 + weight)` (`Gemma3RMSNorm.forward` in transformers; `&self.weight +
    1.0` in the candle reference). Aion's `rmsnorm` op multiplies by gamma
    directly, so the `+1` is folded once here on the host rather than costing an
    op per norm — the same kind of fold as `_mm_b`'s transpose.

    Getting this wrong is silent: the model still runs, and still emits plausible
    embeddings, because RMSNorm's normalization hides the magnitude error.
    """
    return np.ascontiguousarray(w_torch.astype(np.float32, copy=False) + 1.0)


def _f32(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


# ------------------------------- weight structs ------------------------------


# Weights are held as *data*: `nn` layers bind and quantize them themselves, so
# nothing here needs a Builder. Q/K/V and gate/up stay separate, the way the
# checkpoint ships them — pre-concatenating them is a *fusion*, and the compiler
# does that (`opt/fuse_horizontal_matmul` rewrites matmuls sharing an operand
# into one wide matmul plus slices, numerically identically) and then frees the
# sources.


@dataclass
class _LayerWeights:
    input_ln: np.ndarray
    post_attn_ln: np.ndarray
    pre_ffn_ln: np.ndarray
    post_ffn_ln: np.ndarray
    q_proj: np.ndarray
    k_proj: np.ndarray
    v_proj: np.ndarray
    o_proj: np.ndarray
    q_norm: np.ndarray
    k_norm: np.ndarray
    gate_proj: np.ndarray
    up_proj: np.ndarray
    down_proj: np.ndarray


@dataclass
class _SharedWeights:
    embed_tokens: np.ndarray
    final_norm: np.ndarray


def _load_weights(loader: _WeightLoader) -> Tuple[_SharedWeights, List[_LayerWeights]]:
    # Flat key namespace: `embed_tokens.weight`, `layers.{i}.*`, `norm.weight`
    # (a bare `Gemma3TextModel`, with no `model.language_model.` prefix).
    shared = _SharedWeights(
        embed_tokens=_f32(loader.get_f32("embed_tokens.weight")),
        final_norm=_gemma_gamma(loader.get_f32("norm.weight")),
    )

    layers: List[_LayerWeights] = []
    for layer in range(NUM_LAYERS):
        pfx = f"layers.{layer}"
        layers.append(
            _LayerWeights(
                input_ln=_gemma_gamma(loader.get_f32(f"{pfx}.input_layernorm.weight")),
                post_attn_ln=_gemma_gamma(
                    loader.get_f32(f"{pfx}.post_attention_layernorm.weight")
                ),
                pre_ffn_ln=_gemma_gamma(
                    loader.get_f32(f"{pfx}.pre_feedforward_layernorm.weight")
                ),
                post_ffn_ln=_gemma_gamma(
                    loader.get_f32(f"{pfx}.post_feedforward_layernorm.weight")
                ),
                q_proj=_mm_b(loader.get_f32(f"{pfx}.self_attn.q_proj.weight")),
                k_proj=_mm_b(loader.get_f32(f"{pfx}.self_attn.k_proj.weight")),
                v_proj=_mm_b(loader.get_f32(f"{pfx}.self_attn.v_proj.weight")),
                o_proj=_mm_b(loader.get_f32(f"{pfx}.self_attn.o_proj.weight")),
                q_norm=_gemma_gamma(loader.get_f32(f"{pfx}.self_attn.q_norm.weight")),
                k_norm=_gemma_gamma(loader.get_f32(f"{pfx}.self_attn.k_norm.weight")),
                gate_proj=_mm_b(loader.get_f32(f"{pfx}.mlp.gate_proj.weight")),
                up_proj=_mm_b(loader.get_f32(f"{pfx}.mlp.up_proj.weight")),
                down_proj=_mm_b(loader.get_f32(f"{pfx}.mlp.down_proj.weight")),
            )
        )
    return shared, layers


# --------------------- C-ABI Builder authoring helpers -----------------------


def _input(b: Builder, name: str, dtype: aion.AionDType, shape: Shape) -> TensorRef:
    """Declare a public input; string dims are symbolic (dynamic) axes."""
    dims = [_PLACEHOLDER[d] if isinstance(d, str) else int(d) for d in shape]
    dyn = {i: d for i, d in enumerate(shape) if isinstance(d, str)} or None
    return b.input(dims, dtype=dtype, dynamic=dyn).rename(name)


def _rms(gamma: np.ndarray, name: str) -> nn.RMSNorm:
    """A norm with Gemma's epsilon, over an already-folded (`1 + w`) gamma."""
    return nn.RMSNorm(gamma, eps=RMS_EPS, name=name)


# --------------------------------- forward -----------------------------------


def _emit_forward(
    b: Builder,
    shared: _SharedWeights,
    layers: List[_LayerWeights],
    weight_dtype: aion.AionDType,
) -> Tuple[TensorRef, TensorRef]:
    """Emit the encoder. Returns `(embedding, token_embeddings)`."""
    # ---- Public runtime inputs ----
    # The padded tokens and their right-padding mask are the only public inputs.
    # Positions and valid lengths are derived graph values.
    tokens = _input(b, "tokens", aion.int32, ("batch", "seq"))
    attention_mask = _input(b, "attention_mask", aion.int32, ("batch", "seq"))
    positions = b.iota(tokens, axis=1)
    sequence_lengths = b.reduce("sum", attention_mask, axis=1)

    # ---- Token embedding ----
    embed = nn.Embedding(shared.embed_tokens, dtype=weight_dtype, name="embed_tokens")
    # Exact `sqrt(640)`. Upstream rounds this scale to the checkpoint's bf16
    # before multiplying (25.25 instead of 25.2982); the residual stream here is
    # f32, so the exact value is the right one and matches the f32 reference.
    x = F.scale(embed(tokens), math.sqrt(EMBED_DIM))  # [B, S, 640]

    # ---- Transformer blocks ----
    for layer_idx, lw in enumerate(layers):
        # One scope per layer, so every parameter under it is named
        # `layers.{i}/<layer>/<param>` — the key a loaded model addresses it by.
        with b.scope(f"layers.{layer_idx}"):
            # No cache: `attend` below is called without index operands, so k/v
            # are read as a plain sequence — each query masked against its own row,
            # every key visible. GQA (4 query heads over 1 KV head) is the kernel's,
            # so nothing here materializes four copies of a one-head K/V.
            attn = nn.Attention(
                lw.q_proj, lw.o_proj,
                k_proj=lw.k_proj, v_proj=lw.v_proj,
                heads=NUM_HEADS, kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
                scale=ATTN_SCALE, window=aion.AttentionWindow.CAUSAL,
                attn_logits_soft_cap=0.0,
                dtype=weight_dtype, name="self_attn",
            )

            q, k, v = attn.project(_rms(lw.input_ln, "input_layernorm")(x))
            assert k is not None and v is not None

            # Q/K norms and RoPE sit between projection and attention — which is
            # exactly why `project` and `attend` are separate calls. Gemma 3 has
            # no V norm (the checkpoint ships only q_norm/k_norm).
            q = _rms(lw.q_norm, "q_norm")(q)
            k = _rms(lw.k_norm, "k_norm")(k)
            q = b.rope1d(q, positions, base_frequency=ROPE_BASE)
            k = b.rope1d(k, positions, base_frequency=ROPE_BASE)

            o = attn.attend(
                q, k, v,
                query_positions=positions,
                kv_lengths=sequence_lengths,
            )
            x = x + _rms(lw.post_attn_ln, "post_attention_layernorm")(o)

            ff_in = _rms(lw.pre_ffn_ln, "pre_feedforward_layernorm")(x)
            # Split gate/up: the compiler fuses the matmul pair, and the GEGLU is one
            # `gate` op with the tanh gelu approximation (i.e. this model's
            # `gelu_pytorch_tanh`), so there is nothing to pre-concatenate here.
            ff = nn.GatedMLP(lw.gate_proj, lw.up_proj, lw.down_proj,
                             act="gelu", dtype=weight_dtype, name="mlp")(ff_in)
            x = x + _rms(lw.post_ffn_ln, "post_feedforward_layernorm")(ff)

    # ---- Final norm ----
    token_embeddings = _rms(shared.final_norm, "norm")(x)  # [B, S, 640]

    # ---- Last-token pooling ----
    # The pooled position is `sum(attention_mask) - 1`. With right padding this
    # is each row's last real token, normally the `<eos>` appended by tokenizer.
    one = b.cast(b.constant(1.0), aion.int32)
    last_index = b.sub(sequence_lengths, one)                    # [B]
    pooled = b.gather(
        token_embeddings,
        b.unsqueeze(last_index, 1),
        axis=1,
        batch_dims=1,
    )                                                            # [B, 1, 640]
    pooled = b.squeeze(pooled, 1)                                # [B, 640]

    # ---- L2 normalization ----
    # `x / ||x||`, so cosine similarity is a plain dot product. Unsqueezing the
    # per-row norm to `[B,1]` makes ordinary right-aligned broadcasting explicit.
    sum_sq = b.reduce("sum", b.mul(pooled, pooled), axis=1)      # [B]
    row_norm = b.unsqueeze(b.unary("sqrt", sum_sq), axis=1)     # [B, 1]
    embedding = b.div(pooled, row_norm)                          # [B, 640]

    return embedding, token_embeddings


def _finalize(
    b: Builder,
    embedding: TensorRef,
    token_embeddings: TensorRef,
    weight_dtype: aion.AionDType,
) -> None:
    b.mark_output(embedding, "embedding")
    # The pre-pooling hidden states, for callers who want mean pooling or
    # per-token vectors. Already materialized by the final norm, so exposing it
    # costs a copy-out only when it is actually requested.
    b.mark_output(token_embeddings, "token_embeddings")

    quant = "q8_0-weights" if weight_dtype == aion.q8_0 else "f32-weights"
    for key, value in [
        ("arch", "harrier-oss-v1-270m"),
        ("base_arch", "gemma3_text"),
        ("task", "text-embedding"),
        ("quant", quant),
        ("num_layers", str(NUM_LAYERS)),
        ("embed_dim", str(EMBED_DIM)),
        ("num_heads", str(NUM_HEADS)),
        ("num_kv_heads", str(NUM_KV_HEADS)),
        ("head_dim", str(HEAD_DIM)),
        ("ffn_dim", str(FFN_DIM)),
        ("vocab_size", str(VOCAB_SIZE)),
        ("rope_theta", str(ROPE_BASE)),
        ("attention", "causal-full"),
        ("max_position_embeddings", str(MAX_POSITIONS)),
        # How to *use* the outputs. A driver that pools differently, or skips the
        # normalization, silently produces embeddings that do not compare.
        ("pooling", "last-token"),
        ("normalize", "l2"),
        ("similarity_fn", "cosine"),
        ("batch_size", "dynamic"),
        ("padding_side", "right"),
        ("attention_mask", "required"),
        ("stateless", "true"),
        ("bos_token_id", str(BOS_TOKEN_ID)),
        ("eos_token_id", str(EOS_TOKEN_ID)),
        ("pad_token_id", str(PAD_TOKEN_ID)),
        ("tokenizer.add_bos_token", "true"),
        ("tokenizer.add_eos_token", "true"),
        *[(f"prompt.{name}", text) for name, text in PROMPTS.items()],
    ]:
        b.add_metadata(key, value)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    safetensors_path, config_path = _resolve_paths(args.in_path)

    if not args.skip_config_check:
        if config_path is None:
            raise SystemExit(
                f"no config.json next to {safetensors_path}; point at the HF snapshot "
                "directory, or pass --skip-config-check"
            )
        _verify_config(config_path)

    weight_dtype = aion.q8_0 if args.dtype == "q8_0" else aion.float32

    with aion.Context(thread_count=1) as ctx:
        with Builder(ctx) as b:
            with _WeightLoader(safetensors_path) as loader:
                shared, layers = _load_weights(loader)
            embedding, token_embeddings = _emit_forward(b, shared, layers, weight_dtype)
            _finalize(b, embedding, token_embeddings, weight_dtype)
            b.export(os.path.abspath(args.out_aion), None)

    print(f"[harrier-oss-v1] wrote {args.out_aion} ({args.dtype} weights)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
