# SPDX-License-Identifier: Apache-2.0
"""The prebuilt wgpu-native runtime for `aion`'s optional GPU backend.

This package ships a single native library and exposes its path. `aion` imports
this lazily and hands the path to its native loader (via ``AION_WGPU_LIB``); you
normally never call this yourself. Install it with ``pip install aion-engine[gpu]``.
"""
from __future__ import annotations

from pathlib import Path

# Versioned after the wgpu-native release wrapped here (see build.zig.zon).
__version__ = "29.0.4"

_DIR = Path(__file__).resolve().parent
_LIB_NAMES = ("wgpu_native.dll", "libwgpu_native.so", "libwgpu_native.dylib")


def library_path() -> str | None:
    """Absolute path to the bundled wgpu-native library, or None if not present."""
    for name in _LIB_NAMES:
        candidate = _DIR / name
        if candidate.is_file():
            return str(candidate)
    return None


__all__ = ["library_path", "__version__"]
