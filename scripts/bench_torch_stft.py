#!/usr/bin/env python3
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
"""Benchmark PyTorch's STFT to compare against the Aion STFT bench.

The defaults match the NeMo / Parakeet FastConformer streaming front-end
(AudioToMelSpectrogramPreprocessor): 16 kHz, n_fft=512, win_length=400 (25 ms
Hann), hop_length=160 (10 ms), center=True, reflect pad. The mel projection
(80 or 128 filters) is a separate matmul and is NOT included here — this benches
the STFT/FFT only, matching the Aion `program stft f32` line in `src/bench.zig`.

The same FLOP estimate is used so the reported GFLOP/s are directly comparable:

    flops_per_frame = n_fft + 2.5 * n_fft * log2(n_fft)   # window mul + real FFT
    total_flops     = flops_per_frame * batch * num_frames * iters

Realistic streaming buffers (num_frames = 1 + samples/hop, center=True):
    160 ms  -> --samples 2560   (17 frames)
    320 ms  -> --samples 5120   (33 frames)
    640 ms  -> --samples 10240  (65 frames)
    1040 ms -> --samples 16640  (105 frames)
    10 s    -> --samples 160000 (1001 frames)

Run (CPU, single thread, to match Aion's default `--thread-count 1`):

    python scripts/bench_torch_stft.py
    # or, inside the project's uv env (if torch is installed there):
    uv run --project bindings/python python scripts/bench_torch_stft.py

Compare with:

    zig build bench        # look for the "program stft f32 ..." line
"""

from __future__ import annotations

import argparse
import math
import time

import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-fft", type=int, default=512)
    p.add_argument("--win-length", type=int, default=400, help="NeMo/Parakeet uses 400 (25 ms); padded to n_fft")
    p.add_argument("--hop", type=int, default=160)
    p.add_argument("--samples", type=int, default=16000, help="audio buffer length; see header for streaming sizes")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--threads", type=int, default=1, help="torch CPU threads (Aion bench default is 1)")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--no-center", action="store_true", help="disable centered framing (no reflect padding)")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    args = p.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = torch.device(args.device)
    center = not args.no_center

    torch.manual_seed(0)
    signal = torch.randn(args.batch, args.samples, dtype=dtype, device=device)
    window = torch.hann_window(args.win_length, periodic=True, dtype=dtype, device=device)

    def run_once() -> torch.Tensor:
        return torch.stft(
            signal,
            n_fft=args.n_fft,
            hop_length=args.hop,
            win_length=args.win_length,
            window=window,
            center=center,
            pad_mode="reflect",
            return_complex=True,
            onesided=True,
        )

    # Warmup (also resolves lazy init / plan caches).
    for _ in range(args.warmup):
        out = run_once()
    if device.type == "cuda":
        torch.cuda.synchronize()

    num_frames = out.shape[-1]  # [batch, bins, frames]

    t0 = time.perf_counter()
    for _ in range(args.iters):
        out = run_once()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed_s = max(t1 - t0, 1e-12)
    ns = elapsed_s * 1e9
    ms_per_iter = (elapsed_s * 1e3) / args.iters

    log2n = math.log2(args.n_fft)
    flops_per_frame = args.n_fft + 2.5 * args.n_fft * log2n
    total_flops = flops_per_frame * args.batch * num_frames * args.iters
    gflops = total_flops / ns

    chk = float(out.real.flatten()[0].item()) + float(
        out.real.flatten()[min(out.real.numel() - 1, args.n_fft // 2 + 1)].item()
    )

    print(
        f"torch.stft f32 (nfft={args.n_fft} hop={args.hop} frames={num_frames} "
        f"batch={args.batch} threads={torch.get_num_threads()} device={device.type}): "
        f"{gflops:.3f} GFLOP/s  (chk={chk:.4f}, wall={elapsed_s * 1e3:.3f} ms, iter={ms_per_iter:.3f} ms)"
    )
    print(f"  torch={torch.__version__}  win_length={args.win_length}  center={center}  pad_mode=reflect  window=hann(periodic)")


if __name__ == "__main__":
    main()
