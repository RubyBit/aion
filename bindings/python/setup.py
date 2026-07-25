# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    """Build the cffi extension.

    Aion is built (via tools/build_zig.py) and static-linked into the extension
    module by the cffi builder at `src/aion/_ffi/build.py` — that file calls
    `build_zig.build_aion()` when cffi imports it during `super().run()` below.
    This hook pins the Zig install prefix to a deterministic location under
    `build_temp` so repeated builds reuse the same artifacts.

    GPU builds ship no extra runtime here: wgpu-native is loaded at runtime via
    dlopen and delivered by the separate `aion-wgpu` package (see the README).
    """

    def run(self):
        prefix = Path(self.build_temp).resolve() / "aion-zig-prefix"
        prefix.mkdir(parents=True, exist_ok=True)
        os.environ["AION_PY_BUILD_PREFIX"] = str(prefix)

        # Native CPU tuning for local builds unless explicitly overridden.
        os.environ.setdefault("AION_PY_CPU", "native")

        super().run()


# cffi will import the builder referenced below during the build.
setup(
    cffi_modules=["src/aion/_ffi/build.py:ffibuilder"],
    cmdclass={"build_ext": build_ext},
)
