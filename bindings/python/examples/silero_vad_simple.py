# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

import aion

from _common import add_device_args, load_model_with_device


def default_model_path() -> Path:
    # bindings/python/examples -> repo root
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "models" / "silero" / "silero_vad_16k.aion"


def build_signal(chunks: int, num_samples: int, context_size: int) -> np.ndarray:
    total = context_size + (chunks * num_samples)
    signal = np.zeros(total, dtype=np.float32)

    t = np.arange(chunks * num_samples, dtype=np.float32)
    # Simple deterministic synthetic "speech-like" test tone.
    s0 = 0.35 * np.sin(2.0 * math.pi * 220.0 * (t / 16_000.0), dtype=np.float32)
    s1 = 0.20 * np.sin(2.0 * math.pi * 440.0 * (t / 16_000.0), dtype=np.float32)
    n0 = 0.02 * np.sin(2.0 * math.pi * 37.0 * (t / 16_000.0), dtype=np.float32)
    signal[context_size:] = s0 + s1 + n0
    return signal


def main() -> None:
    p = argparse.ArgumentParser(description="Simple Silero VAD streaming example using aion Python bindings")
    p.add_argument("--model", type=Path, default=default_model_path(), help="Path to silero_vad_16k.aion")
    p.add_argument("--chunks", type=int, default=8)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--context-size", type=int, default=64)
    p.add_argument("--bench-iters", type=int, default=1)
    p.add_argument("--thread-count", type=int, default=1, help="Context thread count (default: 1)")
    add_device_args(p)
    p.add_argument("--print-probs", action="store_true")
    args = p.parse_args()

    model_path = args.model.resolve()
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")

    signal = build_signal(args.chunks, args.num_samples, args.context_size)
    chunk_input_len = args.context_size + args.num_samples

    model, device_str = load_model_with_device(str(model_path), args, thread_count=args.thread_count)
    print(f"device: {device_str}")
    with model:
        # Matches the model exported by examples/silero_vad.zig. The LSTM state
        # (h, c) is io-aliased to its outputs (next_h, next_c) in the .aion, so we
        # never bind it: the runtime auto-initializes h/c to zeros on the first run
        # and carries them across chunks by itself. We only bind the real input `x`.
        x_name = "x"
        prob_name = "prob"

        ctx = model.context

        x_t = aion.Tensor.empty(ctx, (1, chunk_input_len), dtype=aion.AionDType.AION_DTYPE_F32)

        prob_t = None
        probs = np.zeros(args.chunks, dtype=np.float32)

        try:
            t0 = time.perf_counter_ns()
            for _ in range(args.bench_iters):
                # Each pass is an independent stream: clear the carried LSTM state.
                model.reset_state()

                for chunk_idx in range(args.chunks):
                    start = chunk_idx * args.num_samples
                    end = start + chunk_input_len
                    x = signal[start:end].reshape(1, chunk_input_len)

                    x_t.write_from_numpy(x)
                    model.bind_input(x_name, x_t)
                    model.run()

                    if prob_t is None:
                        prob_t = model.output_tensor(prob_name)

                    probs[chunk_idx] = prob_t.read_f32()[0]

                    if args.print_probs:
                        t_ms = (chunk_idx * args.num_samples * 1000.0) / 16_000.0
                        print(f"chunk={chunk_idx:03d} t={t_ms:8.1f}ms prob={probs[chunk_idx]:.6f}")

            t1 = time.perf_counter_ns()
        finally:
            if prob_t is not None:
                prob_t.close()
            x_t.close()

    total_chunks = args.chunks * args.bench_iters
    elapsed_ns = max(1, t1 - t0)
    chunks_per_s = (total_chunks * 1_000_000_000.0) / elapsed_ns
    us_per_chunk = elapsed_ns / (total_chunks * 1_000.0)

    print("simple silero example complete")
    print(
        f"chunks={args.chunks} num_samples={args.num_samples} context={args.context_size} "
        f"bench_iters={args.bench_iters}"
    )
    print(
        "probs: "
        f"first={float(probs[0]):.6f} "
        f"last={float(probs[-1]):.6f} "
        f"min={float(np.min(probs)):.6f} "
        f"max={float(np.max(probs)):.6f} "
        f"mean={float(np.mean(probs)):.6f}"
    )
    print(f"perf: total_chunks={total_chunks} chunks/s={chunks_per_s:.2f} us/chunk={us_per_chunk:.2f}")


if __name__ == "__main__":
    main()
