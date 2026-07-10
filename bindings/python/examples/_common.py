# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the example scripts (device selection, etc.).

Run as scripts (`python examples/foo.py`), so the examples directory is on
`sys.path[0]` and `import _common` resolves.
"""
from __future__ import annotations

import argparse

import aion


def add_device_args(p: argparse.ArgumentParser) -> None:
    """Add `--device` / `--gpu-index` / `--gpu-power` to an example parser."""

    p.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="cpu",
        help="Execution device (default: cpu). 'gpu' runs on the GPU (requires a "
        "GPU build: `[tool.aion] gpu=true` / AION_PY_GPU=1). NOTE: GPU model "
        "execution is experimental — full models may hit ops the GPU backend "
        "does not cover yet and raise AION_UNSUPPORTED. 'auto' uses the GPU when "
        "available and silently falls back to CPU if GPU load fails.",
    )
    p.add_argument(
        "--gpu-index",
        type=int,
        default=None,
        help="Physical GPU adapter index (e.g. to choose integrated vs discrete). "
        "Default: pick by --gpu-power.",
    )
    p.add_argument(
        "--gpu-power",
        choices=["default", "low", "high"],
        default="high",
        help="GPU power preference when --gpu-index is unset (high=discrete, low=integrated).",
    )


def load_model_with_device(path: str, args, *, thread_count: int = 1):
    """Load `path` honoring `--device`/`--gpu-index`/`--gpu-power`.

    Returns ``(model, device_str)``. With `--device auto`, a GPU failure
    (CPU-only build or no adapter) falls back to CPU with a printed notice;
    with `--device gpu` the failure propagates.
    """

    want = getattr(args, "device", "cpu")
    gpu_index = getattr(args, "gpu_index", None)
    gpu_power = getattr(args, "gpu_power", "high")

    if want == "cpu":
        return aion.load_model(path, thread_count=thread_count), "cpu"

    try:
        model = aion.load_model(
            path,
            thread_count=thread_count,
            device="gpu",
            adapter_index=gpu_index,
            power=gpu_power,
        )
        dev = f"gpu:{gpu_index}" if gpu_index is not None else "gpu"
        return model, dev
    except aion.AionError as e:
        if want == "gpu":
            raise
        print(f"[device] GPU unavailable ({e}); falling back to CPU.")
        return aion.load_model(path, thread_count=thread_count), "cpu"
