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
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import aion


def build_signal(chunks: int, num_samples: int, context_size: int) -> np.ndarray:
    total = context_size + (chunks * num_samples)
    signal = np.zeros(total, dtype=np.float32)

    t = np.arange(chunks * num_samples, dtype=np.float32)
    s0 = 0.35 * np.sin(2.0 * math.pi * 220.0 * (t / 16_000.0), dtype=np.float32)
    s1 = 0.20 * np.sin(2.0 * math.pi * 440.0 * (t / 16_000.0), dtype=np.float32)
    n0 = 0.02 * np.sin(2.0 * math.pi * 37.0 * (t / 16_000.0), dtype=np.float32)
    signal[context_size:] = s0 + s1 + n0
    return signal


@dataclass
class RunResult:
    probs: np.ndarray


def run_aion(model_path: Path, *, chunks: int, num_samples: int, context_size: int, thread_count: int, print_probs: bool) -> RunResult:
    signal = build_signal(chunks, num_samples, context_size)
    chunk_input_len = context_size + num_samples

    with aion.load_model(str(model_path), thread_count=thread_count) as model:
        ctx = model.context

        x_t = aion.Tensor.empty(ctx, (1, chunk_input_len), dtype=aion.AionDType.AION_DTYPE_F32)
        h_t = aion.Tensor.zeros((1, 128), ctx=ctx, dtype=aion.AionDType.AION_DTYPE_F32)
        c_t = aion.Tensor.zeros((1, 128), ctx=ctx, dtype=aion.AionDType.AION_DTYPE_F32)

        prob_t = None
        next_h_t = None
        next_c_t = None

        probs = np.zeros(chunks, dtype=np.float32)

        try:
            h_t.zero()
            c_t.zero()
            h_in = h_t
            c_in = c_t

            for chunk_idx in range(chunks):
                start = chunk_idx * num_samples
                end = start + chunk_input_len
                x = signal[start:end].reshape(1, chunk_input_len)

                x_t.copy_from(x)
                model.bind_input("x", x_t)
                model.bind_input("h", h_in)
                model.bind_input("c", c_in)
                model.run()

                if prob_t is None:
                    prob_t = model.output_tensor("prob")
                    next_h_t = model.output_tensor("next_h")
                    next_c_t = model.output_tensor("next_c")

                assert next_h_t is not None and next_c_t is not None  # fetched with prob_t above
                probs[chunk_idx] = float(prob_t.numpy().reshape(-1)[0])
                h_in = next_h_t
                c_in = next_c_t

                if print_probs:
                    t_ms = (chunk_idx * num_samples * 1000.0) / 16_000.0
                    print(f"aion    chunk={chunk_idx:03d} t={t_ms:8.1f}ms prob={probs[chunk_idx]:.6f}")

        finally:
            if prob_t is not None:
                prob_t.close()
            if next_h_t is not None:
                next_h_t.close()
            if next_c_t is not None:
                next_c_t.close()
            x_t.close()
            h_t.close()
            c_t.close()

    return RunResult(probs=probs)


def _load_tinygrad_model(safetensors_path: Path, *, stft_pad_mode: str):
    # Import tinygrad lazily so this script can still run in environments
    # without it installed.
    from tinygrad import Tensor  # noqa: F401
    from tinygrad import nn
    from tinygrad.nn.state import load_state_dict, safe_load

    class TinySileroVAD:
        def __init__(self):
            self.n_fft = 256
            self.stride = 128
            self.pad = 64
            self.cutoff = int(self.n_fft // 2) + 1

            self.stft_conv = nn.Conv1d(1, 258, kernel_size=256, stride=self.stride, padding=0, bias=False)
            self.conv1 = nn.Conv1d(129, 128, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv1d(128, 64, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
            self.conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)

            self.lstm_cell = nn.LSTMCell(128, 128)
            self.final_conv = nn.Conv1d(128, 1, 1)

        def __call__(self, x, state=None):
            if state is not None:
                state = (state[0], state[1])

            x = x.pad((0, self.pad), stft_pad_mode).unsqueeze(1)
            x = self.stft_conv(x)
            x = (x[:, : self.cutoff, :] ** 2 + x[:, self.cutoff :, :] ** 2).sqrt()
            x = self.conv1(x).relu()
            x = self.conv2(x).relu()
            x = self.conv3(x).relu()
            x = self.conv4(x).relu().squeeze(-1)

            h, c = self.lstm_cell(x, state)
            x = h.unsqueeze(-1)
            state = h.stack(c, dim=0)

            x = x.relu()
            x = self.final_conv(x).sigmoid()
            x = x.squeeze(1).mean(axis=1).unsqueeze(1)
            return x, state

    model = TinySileroVAD()
    state_dict = safe_load(str(safetensors_path))
    load_state_dict(model, state_dict)
    return model


def run_tinygrad(
    safetensors_path: Path,
    *,
    chunks: int,
    num_samples: int,
    context_size: int,
    stft_pad_mode: str,
    print_probs: bool,
) -> RunResult:
    from tinygrad import Tensor

    signal = build_signal(chunks, num_samples, context_size)
    chunk_input_len = context_size + num_samples

    model = _load_tinygrad_model(safetensors_path, stft_pad_mode=stft_pad_mode)

    probs = np.zeros(chunks, dtype=np.float32)
    state = None

    for chunk_idx in range(chunks):
        start = chunk_idx * num_samples
        end = start + chunk_input_len
        x_np = signal[start:end].reshape(1, chunk_input_len)
        x = Tensor(x_np).float()

        out, state = model(x, state)

        # out is [1, 1]
        out_np = np.array(out.numpy(), dtype=np.float32)
        probs[chunk_idx] = float(out_np.reshape(-1)[0])

        if print_probs:
            t_ms = (chunk_idx * num_samples * 1000.0) / 16_000.0
            print(f"tinygrad chunk={chunk_idx:03d} t={t_ms:8.1f}ms prob={probs[chunk_idx]:.6f}")

    return RunResult(probs=probs)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare AION Silero VAD output vs Tinygrad reference")
    ap.add_argument("--aion-model", type=Path, default=Path("models/silero/silero_vad_16k.aion"))
    ap.add_argument("--safetensors", type=Path, default=Path("models/silero/silero_vad_16k.safetensors"))
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--num-samples", type=int, default=512)
    ap.add_argument("--context-size", type=int, default=64)
    ap.add_argument("--thread-count", type=int, default=1)
    ap.add_argument(
        "--tinygrad-stft-pad-mode",
        type=str,
        default="reflect",
        choices=("reflect", "constant", "replicate"),
        help="Pad mode used by tinygrad before stft_conv (default: reflect)",
    )
    ap.add_argument("--print-probs", action="store_true")
    args = ap.parse_args()

    if not args.aion_model.is_file():
        raise SystemExit(f"AION model not found: {args.aion_model}")
    if not args.safetensors.is_file():
        raise SystemExit(f"safetensors not found: {args.safetensors}")

    aion_res = run_aion(
        args.aion_model,
        chunks=args.chunks,
        num_samples=args.num_samples,
        context_size=args.context_size,
        thread_count=args.thread_count,
        print_probs=args.print_probs,
    )

    tg_res = run_tinygrad(
        args.safetensors,
        chunks=args.chunks,
        num_samples=args.num_samples,
        context_size=args.context_size,
        stft_pad_mode=args.tinygrad_stft_pad_mode,
        print_probs=args.print_probs,
    )

    if aion_res.probs.shape != tg_res.probs.shape:
        raise SystemExit(f"shape mismatch: aion {aion_res.probs.shape} vs tinygrad {tg_res.probs.shape}")

    diff = aion_res.probs - tg_res.probs
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))

    print("\nsummary")
    print(f"  max_abs_diff={max_abs:.6g}  mean_abs_diff={mean_abs:.6g}")
    print(
        "  aion:    first={:.6f} last={:.6f} min={:.6f} max={:.6f} mean={:.6f}".format(
            float(aion_res.probs[0]),
            float(aion_res.probs[-1]),
            float(np.min(aion_res.probs)),
            float(np.max(aion_res.probs)),
            float(np.mean(aion_res.probs)),
        )
    )
    print(
        "  tinygrad: first={:.6f} last={:.6f} min={:.6f} max={:.6f} mean={:.6f}".format(
            float(tg_res.probs[0]),
            float(tg_res.probs[-1]),
            float(np.min(tg_res.probs)),
            float(np.max(tg_res.probs)),
            float(np.mean(tg_res.probs)),
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
