# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the example scripts (device selection, etc.).

Run as scripts (`python examples/foo.py`), so the examples directory is on
`sys.path[0]` and `import _common` resolves.
"""
from __future__ import annotations

import argparse
import struct

import aion


def read_aion_metadata(path: str) -> dict:
    """Read a `.aion` package's key/value metadata section without loading it.

    Converters record model parameters here (arch, chunk size, window sizes,
    ...), so examples can auto-configure from the model file alone. Layout per
    aion._writer.format: 72-byte header (magic, version, section count,
    reserved, directory offset), a directory of (type u32, flags u32, offset
    u64, size u64) entries, then section payloads. Strings section (type 1) is
    a u32 count + length-prefixed strings; metadata section (type 9) is a u32
    count + (key, value) string-table index pairs.
    """
    with open(path, "rb") as fh:
        header = fh.read(72)
        if len(header) < 72 or header[:4] != b"AION":
            raise ValueError(f"{path}: not a .aion package")
        section_count = struct.unpack_from("<I", header, 8)[0]
        dir_offset = struct.unpack_from("<Q", header, 16)[0]
        fh.seek(dir_offset)
        directory = fh.read(section_count * 24)
        sections = {}
        for i in range(section_count):
            stype, _flags, off, size = struct.unpack_from("<IIQQ", directory, i * 24)
            sections[stype] = (off, size)
        if 9 not in sections or 1 not in sections:  # metadata / strings
            return {}

        def section_bytes(stype: int) -> bytes:
            off, size = sections[stype]
            fh.seek(off)
            return fh.read(size)

        sb = section_bytes(1)
        (n_str,) = struct.unpack_from("<I", sb, 0)
        pos, strings = 4, []
        for _ in range(n_str):
            (ln,) = struct.unpack_from("<I", sb, pos)
            pos += 4
            strings.append(sb[pos:pos + ln].decode("utf-8"))
            pos += ln
        mb = section_bytes(9)
        (n_md,) = struct.unpack_from("<I", mb, 0)
        return {
            strings[struct.unpack_from("<I", mb, 4 + i * 8)[0]]:
            strings[struct.unpack_from("<I", mb, 8 + i * 8)[0]]
            for i in range(n_md)
        }


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


def load_model_with_device(path: str, args, *, thread_count: int = 1, **load_kwargs):
    """Load `path` honoring `--device`/`--gpu-index`/`--gpu-power`.

    Extra keyword args (e.g. `cache_capacity`, `growable`,
    `initial_cache_capacity`, `auto_positions`) are forwarded to
    `aion.load_model`. Returns ``(model, device_str)``. With `--device auto`, a
    GPU failure (CPU-only build or no adapter) falls back to CPU with a printed
    notice; with `--device gpu` the failure propagates.
    """

    want = getattr(args, "device", "cpu")
    gpu_index = getattr(args, "gpu_index", None)
    gpu_power = getattr(args, "gpu_power", "high")

    if want == "cpu":
        return aion.load_model(path, thread_count=thread_count, **load_kwargs), "cpu"

    try:
        model = aion.load_model(
            path,
            thread_count=thread_count,
            device="gpu",
            adapter_index=gpu_index,
            power=gpu_power,
            **load_kwargs,
        )
        dev = f"gpu:{gpu_index}" if gpu_index is not None else "gpu"
        return model, dev
    except aion.AionError as e:
        if want == "gpu":
            raise
        print(f"[device] GPU unavailable ({e}); falling back to CPU.")
        return aion.load_model(path, thread_count=thread_count, **load_kwargs), "cpu"
