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
- Serialization is delegated to `scripts/_aion_writer.py` (shared with Gemma).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _aion_writer as aw  # type: ignore  # noqa: E402


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


def build_tiny_silero_aion_package(
    weights: Dict[str, np.ndarray],
    num_samples: int,
    context_size: int,
    pad_mode: int,
    stft_pad_right: int,
) -> aw.Package:
    # Model constants (match examples/silero_vad.zig)
    n_fft = 256
    stride = 128
    cutoff = (n_fft // 2) + 1

    chunk_input_len = int(num_samples + context_size)
    if chunk_input_len < n_fft:
        raise ValueError("chunk_input_len must be >= 256")

    if stft_pad_right < 0:
        raise ValueError("stft_pad_right must be >= 0")

    padded_len = chunk_input_len + int(stft_pad_right)
    numer = padded_len - n_fft
    t_steps = (numer // stride) + 1
    if t_steps <= 0:
        raise ValueError("invalid t_steps")

    # Symbolic batch.
    dim_symbols = ["batch"]
    # dim_exprs: one expr 0 = symbol 0.
    dim_exprs_symbol_indices = [0]
    batch_expr_index = 0

    initializers: List[aw.Initializer] = []
    values: List[aw.ValueRecord] = []
    nodes: List[aw.NodeRecord] = []
    debug_names: List[aw.DebugName] = []

    # Keep rank bookkeeping just for sanity / signature correctness.
    ranks: Dict[int, int] = {}

    def add_public_input(name: str, shape: Tuple[int, ...], batch_symbol: bool) -> int:
        vid = len(values)
        st = aw.shape_terms_for(shape, batch_expr_index if batch_symbol else None)
        values.append(
            aw.ValueRecord(
                dtype=aw.DType.f32,
                rank=len(shape),
                source=aw.ValueSource.public_input,
                shape_terms=st,
                initializer_index=None,
            )
        )
        ranks[vid] = len(shape)
        debug_names.append(aw.DebugName(value=vid, name=name))
        return vid

    def add_initializer_value(name: str, array: np.ndarray) -> int:
        nonlocal initializers
        arr = np.ascontiguousarray(array.astype(np.float32, copy=False))
        data = arr.astype("<f4", copy=False).tobytes(order="C")
        init_idx = len(initializers)
        initializers.append(aw.Initializer.plain(aw.DType.f32, data))

        vid = len(values)
        st = aw.shape_terms_for(tuple(int(x) for x in arr.shape))
        values.append(
            aw.ValueRecord(
                dtype=aw.DType.f32,
                rank=arr.ndim,
                source=aw.ValueSource.initializer,
                shape_terms=st,
                initializer_index=init_idx,
            )
        )
        ranks[vid] = arr.ndim
        debug_names.append(aw.DebugName(value=vid, name=name))
        return vid

    def add_produced_value(rank: int) -> int:
        vid = len(values)
        values.append(
            aw.ValueRecord(
                dtype=aw.DType.f32,
                rank=rank,
                source=aw.ValueSource.produced,
                shape_terms=aw.u32(0),
                initializer_index=None,
            )
        )
        ranks[vid] = rank
        return vid

    def emit(kind: int, inputs: List[int], attr: bytes, out_rank: int) -> int:
        out_vid = add_produced_value(out_rank)
        nodes.append(aw.NodeRecord(kind=kind, output=out_vid, inputs=inputs, attr=attr))
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
    x3 = emit(aw.NodeKind.ViewUnsqueeze, [x], aw.attr_view_unsqueeze(2), out_rank=3)

    stft = emit(
        aw.NodeKind.Conv1D,
        [x3, stft_w],
        aw.attr_conv1d(
            stride=128,
            dilation=1,
            pad_left=0,
            pad_right=int(stft_pad_right),
            pad_mode=pad_mode,
            groups=1,
        ),
        out_rank=3,
    )

    # real/imag slices with dynamic batch lens via shape terms.
    lens3 = [aw.shape_term_expr(batch_expr_index), aw.shape_term_constant(t_steps), aw.shape_term_constant(cutoff)]
    real = emit(aw.NodeKind.ViewSliceND, [stft], aw.attr_view_slice_nd([0, 0, 0], lens3), out_rank=3)
    imag = emit(aw.NodeKind.ViewSliceND, [stft], aw.attr_view_slice_nd([0, 0, cutoff], lens3), out_rank=3)

    r2 = emit(aw.NodeKind.ElemwiseBinary, [real, real], aw.attr_binary(aw.ElemwiseBinaryOp.mul), out_rank=3)
    im2 = emit(aw.NodeKind.ElemwiseBinary, [imag, imag], aw.attr_binary(aw.ElemwiseBinaryOp.mul), out_rank=3)
    mag2 = emit(aw.NodeKind.ElemwiseBinary, [r2, im2], aw.attr_binary(aw.ElemwiseBinaryOp.add), out_rank=3)
    mag = emit(aw.NodeKind.Unary, [mag2], aw.attr_unary(aw.UnaryOp.sqrt), out_rank=3)

    conv1_out = emit(
        aw.NodeKind.Conv1D,
        [mag, conv1_w, conv1_b],
        aw.attr_conv1d(stride=1, dilation=1, pad_left=1, pad_right=1, pad_mode=aw.PadMode.zero, groups=1),
        out_rank=3,
    )
    c1 = emit(aw.NodeKind.Unary, [conv1_out], aw.attr_unary(aw.UnaryOp.relu), out_rank=3)

    conv2_out = emit(
        aw.NodeKind.Conv1D,
        [c1, conv2_w, conv2_b],
        aw.attr_conv1d(stride=2, dilation=1, pad_left=1, pad_right=1, pad_mode=aw.PadMode.zero, groups=1),
        out_rank=3,
    )
    c2 = emit(aw.NodeKind.Unary, [conv2_out], aw.attr_unary(aw.UnaryOp.relu), out_rank=3)

    conv3_out = emit(
        aw.NodeKind.Conv1D,
        [c2, conv3_w, conv3_b],
        aw.attr_conv1d(stride=2, dilation=1, pad_left=1, pad_right=1, pad_mode=aw.PadMode.zero, groups=1),
        out_rank=3,
    )
    c3 = emit(aw.NodeKind.Unary, [conv3_out], aw.attr_unary(aw.UnaryOp.relu), out_rank=3)

    conv4_out = emit(
        aw.NodeKind.Conv1D,
        [c3, conv4_w, conv4_b],
        aw.attr_conv1d(stride=1, dilation=1, pad_left=1, pad_right=1, pad_mode=aw.PadMode.zero, groups=1),
        out_rank=3,
    )
    c4 = emit(aw.NodeKind.Unary, [conv4_out], aw.attr_unary(aw.UnaryOp.relu), out_rank=3)

    features = emit(aw.NodeKind.ViewSqueeze, [c4], aw.attr_view_squeeze(1), out_rank=2)

    packed_state = emit(
        aw.NodeKind.LSTMCell,
        [features, h, c, lstm_w_ih, lstm_w_hh, lstm_b_ih, lstm_b_hh],
        aw.attr_lstm(True),
        out_rank=2,
    )

    # Slice packed state [batch, 2*hidden] into h and c.
    lens2_h = [aw.shape_term_expr(batch_expr_index), aw.shape_term_constant(128)]
    h_t = emit(aw.NodeKind.ViewSliceND, [packed_state], aw.attr_view_slice_nd([0, 0], lens2_h), out_rank=2)
    c_t = emit(aw.NodeKind.ViewSliceND, [packed_state], aw.attr_view_slice_nd([0, 128], lens2_h), out_rank=2)

    h3 = emit(aw.NodeKind.ViewUnsqueeze, [h_t], aw.attr_view_unsqueeze(1), out_rank=3)
    act = emit(aw.NodeKind.Unary, [h3], aw.attr_unary(aw.UnaryOp.relu), out_rank=3)

    logits3 = emit(
        aw.NodeKind.Conv1D,
        [act, final_w, final_b],
        aw.attr_conv1d(stride=1, dilation=1, pad_left=0, pad_right=0, pad_mode=aw.PadMode.zero, groups=1),
        out_rank=3,
    )

    probs3 = emit(aw.NodeKind.Unary, [logits3], aw.attr_unary(aw.UnaryOp.sigmoid), out_rank=3)
    probs2 = emit(aw.NodeKind.ViewSqueeze, [probs3], aw.attr_view_squeeze(2), out_rank=2)
    probs1 = emit(aw.NodeKind.Reduce, [probs2], aw.attr_reduce(aw.ReduceOp.mean, axis=1), out_rank=1)
    probs = emit(aw.NodeKind.ViewUnsqueeze, [probs1], aw.attr_view_unsqueeze(1), out_rank=2)

    # Signatures.
    inputs = [aw.NamedValue("x", x), aw.NamedValue("h", h), aw.NamedValue("c", c)]
    outputs = [aw.NamedValue("prob", probs), aw.NamedValue("next_h", h_t), aw.NamedValue("next_c", c_t)]

    # I/O aliases: h -> next_h, c -> next_c (indices within signatures).
    io_aliases = [aw.IoAlias(input_index=1, output_index=1), aw.IoAlias(input_index=2, output_index=2)]

    metadata = [aw.MetadataEntry(key="arch", value="tiny-silero-vad")]

    return aw.Package(
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
    pad_mode = aw.PadMode.reflect if args.pad_mode == "reflect" else aw.PadMode.zero

    pkg = build_tiny_silero_aion_package(
        w,
        num_samples=args.num_samples,
        context_size=args.context,
        pad_mode=pad_mode,
        stft_pad_right=args.stft_pad_right,
    )
    aw.write_aion_v4(args.out_aion, pkg)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))