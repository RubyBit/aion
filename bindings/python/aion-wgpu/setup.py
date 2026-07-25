# SPDX-License-Identifier: Apache-2.0
"""Build backend glue for the `aion-wgpu` runtime package.

Two responsibilities:

1. **Stage** the wgpu-native dynamic library into ``src/aion_wgpu/`` so it ships
   as package data. CI cross-builds copy the library in beforehand (and set
   ``AION_WGPU_STAGED=1``); a plain local/native build instead runs
   ``zig build wgpu-runtime`` — which fetches the library pinned in the repo's
   ``build.zig.zon`` (the single source of truth for the wgpu-native version).

2. **Tag** the wheel as ``py3-none-<platform>``: it contains only a native
   library, so it is independent of the Python version/ABI but specific to the
   platform. ``--plat-name`` (set by CI per target) selects the platform.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # older setuptools
    try:
        from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    except ImportError:
        _bdist_wheel = None

HERE = Path(__file__).resolve().parent
PKG_DIR = HERE / "src" / "aion_wgpu"
REPO_ROOT = HERE.parents[2]  # bindings/python/aion-wgpu -> repo root (has build.zig)

# The wgpu-native dynamic library filename per platform (as installed by
# `zig build wgpu-runtime`).
_LIB_BY_SYSTEM = {
    "windows": "wgpu_native.dll",
    "linux": "libwgpu_native.so",
    "darwin": "libwgpu_native.dylib",
}


def _staged_lib() -> Path | None:
    for name in _LIB_BY_SYSTEM.values():
        candidate = PKG_DIR / name
        if candidate.is_file():
            return candidate
    return None


def _stage_via_zig() -> None:
    """Stage the wgpu-native runtime for the *native* target into the package."""
    zig = os.environ.get("AION_ZIG_EXE", "zig")
    prefix = HERE / "build" / "wgpu-prefix"
    prefix.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [zig, "build", "wgpu-runtime", "-Dgpu=true", "--prefix", str(prefix)],
        cwd=str(REPO_ROOT),
    )
    name = _LIB_BY_SYSTEM[platform.system().lower()]
    src = prefix / name
    if not src.is_file():
        raise RuntimeError(f"zig did not stage the expected runtime library: {src}")
    PKG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, PKG_DIR / name)


class build_py(_build_py):
    def run(self):
        if _staged_lib() is None and not os.environ.get("AION_WGPU_STAGED"):
            _stage_via_zig()
        if _staged_lib() is None:
            raise RuntimeError(
                "aion-wgpu: no wgpu-native runtime staged in src/aion_wgpu/ "
                "(expected one of: " + ", ".join(_LIB_BY_SYSTEM.values()) + ")"
            )
        super().run()


cmdclass: dict = {"build_py": build_py}

if _bdist_wheel is not None:

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            # Carries a native library -> platform-specific wheel, not purelib.
            # (Must be set *after* super(), which otherwise recomputes it.)
            self.root_is_pure = False

        def get_tag(self):
            _python, _abi, plat = super().get_tag()
            # Any CPython 3, no Python ABI dependency, this (native) platform only.
            return "py3", "none", plat

    cmdclass["bdist_wheel"] = bdist_wheel


setup(cmdclass=cmdclass)
