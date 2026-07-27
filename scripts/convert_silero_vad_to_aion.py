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
- Authored via the C-ABI `aion.Builder`; serialization is done by the Zig core
  through `Builder.export()` (no pure-Python writer).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import aion
from aion import Builder, nn
from aion.enums import AionInputRoleKind


def _load_safetensors_numpy(path: str) -> Dict[str, np.ndarray]:
    try:
        from safetensors.numpy import load_file  # type: ignore

        return load_file(path)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency. Install with: pip install safetensors numpy\n"
            f"Import error: {e}"
        )


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


# Every parameter is exported under a `<layer>/<param>` debug name, which is how a
# consumer finds it again (see `examples/silero_vad.zig`). The table that used to map
# those out by hand is gone: the `nn` layer named `conv1` writes `conv1/weight` and
# `conv1/bias` itself, so the names cannot drift from the layers that produce them.


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


# ----------------------------- Graph construction ----------------------------


def build_silero(
    b: Builder,
    weights: Dict[str, np.ndarray],
    num_samples: int,
    context_size: int,
    pad_mode: str,
    stft_pad_right: int,
) -> None:
    """Author the Silero VAD graph on `b` and mark its outputs/roles/aliases.

    Matches examples/silero_vad.zig: NLC Conv1D (weight [k, cin, cout]), a symbolic
    batch axis, and io-aliased zero-init LSTM state (h/c)."""
    n_fft = 256
    stride = 128
    cutoff = (n_fft // 2) + 1

    chunk_input_len = int(num_samples + context_size)
    if chunk_input_len < n_fft:
        raise ValueError("chunk_input_len must be >= 256")
    if stft_pad_right < 0:
        raise ValueError("stft_pad_right must be >= 0")

    padded_len = chunk_input_len + int(stft_pad_right)
    t_steps = ((padded_len - n_fft) // stride) + 1
    if t_steps <= 0:
        raise ValueError("invalid t_steps")

    # Inputs (axis 0 is the symbolic batch; placeholder size 1).
    x = b.input((1, chunk_input_len), dynamic={0: "batch"}).rename("x")
    h = b.input((1, 128), dynamic={0: "batch"}).rename("h")
    c = b.input((1, 128), dynamic={0: "batch"}).rename("c")

    # `nn` layers hold the weights as data and bind them under `<layer>/<param>`
    # themselves, which is exactly the naming this model already exported — so those
    # debug names are now produced by the layer rather than pinned on by hand
    # afterwards, and `examples/silero_vad.zig` finds them unchanged.
    w = weights
    stft_conv = nn.Conv1D(w["stft_w"], name="stft_conv", stride=128,
                          pad_right=int(stft_pad_right), pad_mode=pad_mode)
    convs = [
        nn.Conv1D(w[f"conv{i}_w"], w[f"conv{i}_b"], name=f"conv{i}",
                  stride=stride, pad_left=1, pad_right=1)
        for i, stride in ((1, 1), (2, 2), (3, 2), (4, 1))
    ]
    lstm = nn.LSTMCell(w["lstm_w_ih"], w["lstm_w_hh"], w["lstm_b_ih"], w["lstm_b_hh"],
                       name="lstm")
    final_conv = nn.Conv1D(w["final_w"], w["final_b"], name="final_conv")

    # Forward graph.
    stft = stft_conv(b.unsqueeze(x, 2))

    # real/imag halves; the batch axis length is symbolic.
    real = b.slice(stft, (0, 0, 0), ("batch", t_steps, cutoff))
    imag = b.slice(stft, (0, 0, cutoff), ("batch", t_steps, cutoff))
    mag = b.unary("sqrt", real * real + imag * imag)

    for conv in convs:
        mag = conv(mag).relu()

    features = b.squeeze(mag, 1)  # [batch, 128]

    # `LSTMCell` unpacks the fused op's `[batch, 2h]` state, so the offsets are not
    # the caller's to know.
    h_t, c_t = lstm(features, h, c)
    h_t = h_t.rename("next_h")
    c_t = c_t.rename("next_c")

    probs = final_conv(b.unsqueeze(h_t, 1).relu()).sigmoid()
    probs = b.squeeze(probs, 2)               # [batch, 1]
    probs = b.reduce("mean", probs, axis=1)   # [batch]
    probs = b.unsqueeze(probs, 1).rename("prob")  # [batch, 1]

    b.mark_output(probs, "prob")
    b.mark_output(h_t, "next_h")
    b.mark_output(c_t, "next_c")

    # io-aliased zero-init recurrent LSTM state.
    b.add_input_role(h, AionInputRoleKind.AION_ROLE_STATE, zero_init=True)
    b.add_input_role(c, AionInputRoleKind.AION_ROLE_STATE, zero_init=True)
    b.add_output_alias(h, h_t)
    b.add_output_alias(c, c_t)

    b.add_metadata("arch", "tiny-silero-vad")


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description="Convert silero VAD safetensors to Aion .aion")
    ap.add_argument("in_safetensors", type=str)
    ap.add_argument("out_aion", type=str)
    ap.add_argument("--num-samples", type=int, default=512)
    ap.add_argument("--context", type=int, default=64)
    ap.add_argument("--pad-mode", type=str, default="reflect", choices=("reflect", "zero"))
    ap.add_argument(
        "--stft-pad-right",
        type=int,
        default=64,
        help="Right padding (in samples) applied to the STFT conv input (default: 64)",
    )
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

    with aion.Context(thread_count=1) as ctx:
        with Builder(ctx) as b:
            build_silero(
                b,
                w,
                num_samples=args.num_samples,
                context_size=args.context,
                pad_mode=args.pad_mode,
                stft_pad_right=args.stft_pad_right,
            )
            b.export(os.path.abspath(args.out_aion), None)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))