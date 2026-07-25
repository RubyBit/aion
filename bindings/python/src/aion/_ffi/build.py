# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import sys
from pathlib import Path

from cffi import FFI


def _import_build_zig():
    # This file lives at: bindings/python/src/aion/_ffi/build.py
    # tools/build_zig.py is at: bindings/python/tools/build_zig.py
    python_root = Path(__file__).resolve().parents[3]
    tools_dir = python_root / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import build_zig  # type: ignore
    finally:
        # Keep sys.path clean-ish.
        try:
            sys.path.remove(str(tools_dir))
        except ValueError:
            pass
    return build_zig


def _load_cdef_text() -> str:
    # We keep the FFI declarations in a minimal macro-free header.
    h_path = Path(__file__).with_name("aion_ffi.h")
    return h_path.read_text(encoding="utf-8")


ffibuilder = FFI()
ffibuilder.cdef(_load_cdef_text())

# Build Aion and link the resulting library into the extension.
_build_zig = _import_build_zig()
_artifacts = _build_zig.build_aion()

include_dirs = [str(_artifacts.include_dir)]
extra_objects = [str(_artifacts.link_lib_path)]

# GPU builds add nothing to the link: the `aion` archive references zero wgpu
# symbols and loads wgpu-native at runtime via dlopen (shipped by the optional
# `aion-wgpu` package). So the extension is identical whether or not GPU is on,
# apart from the seam code compiled into the static archive.

extra_compile_args: list[str] = []
extra_link_args: list[str] = []
libraries: list[str] = []

# Linux/macOS: we link a static library into a shared module; ensure pthread flags.
if os.name == "posix":
    extra_compile_args += ["-pthread"]
    extra_link_args += ["-pthread"]
    # The GPU seam resolves wgpu-native at runtime via dlopen/dlsym; on older
    # glibc (e.g. manylinux2014) those live in libdl, so link it explicitly.
    # On macOS they are in libSystem (no -ldl), so only do this on Linux.
    if sys.platform.startswith("linux"):
        libraries += ["dl"]

# Windows: make sure the extension links the basic system import libraries.
# (Some toolchain configurations do not pull these in implicitly for extension modules.)
if os.name == "nt":
    libraries += [
        "kernel32",
        "ntdll",
        # Be explicit about CRT import libraries; some build environments for
        # extension modules end up not pulling these in as expected.
        "ucrt",
        "vcruntime",
    ]

ffibuilder.set_source(
    "aion._aion_cffi",
    '#include <stdint.h>\n#include <stddef.h>\n#include "aion.h"\n',
    include_dirs=include_dirs,
    extra_objects=extra_objects,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    libraries=libraries,
)


if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
