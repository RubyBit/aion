# SPDX-License-Identifier: Apache-2.0
"""Discovery and diagnostics for the optional wgpu-native runtime."""
from __future__ import annotations

import importlib
import importlib.util
import os


def register_wgpu_runtime() -> None:
    """Point the native loader at the companion package's runtime, if present."""
    if os.environ.get("AION_WGPU_LIB"):
        return
    try:
        path = importlib.import_module("aion_wgpu").library_path()
    except Exception:
        return
    if path:
        os.environ["AION_WGPU_LIB"] = str(path)


def missing_wgpu_runtime_hint() -> str | None:
    """Return an actionable hint when no explicit/packaged runtime is available."""
    if os.environ.get("AION_WGPU_LIB"):
        return None
    try:
        companion_installed = importlib.util.find_spec("aion_wgpu") is not None
    except (ImportError, AttributeError, ValueError):
        companion_installed = False
    if companion_installed:
        return (
            "The installed aion-wgpu package did not provide a wgpu-native "
            "library; reinstall it with `uv sync --reinstall-package aion-wgpu`."
        )
    return (
        "The optional wgpu-native runtime is not installed. Run `uv sync` in "
        "bindings/python (development checkout) or install `aion-engine[gpu]`."
    )


__all__ = ["missing_wgpu_runtime_hint", "register_wgpu_runtime"]
