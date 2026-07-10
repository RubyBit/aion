# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    """Build the cffi extension.

    Aion is built (via tools/build_zig.py) and static-linked into the extension
    module by the cffi builder at `src/aion/_ffi/build.py` — that file calls
    `build_zig.build_aion()` when cffi imports it during `super().run()` below.
    This hook pins the Zig install prefix to a deterministic location under
    `build_temp` so repeated builds reuse the same artifacts, and (for GPU builds)
    copies any runtime DLLs next to the built extension so it loads at runtime.
    """

    def run(self):
        prefix = Path(self.build_temp).resolve() / "aion-zig-prefix"
        prefix.mkdir(parents=True, exist_ok=True)
        os.environ["AION_PY_BUILD_PREFIX"] = str(prefix)

        # Native CPU tuning for local builds unless explicitly overridden.
        os.environ.setdefault("AION_PY_CPU", "native")

        super().run()

        # GPU builds: the cffi builder (src/aion/_ffi/build.py) sets this to the
        # wgpu runtime DLL(s). Place them beside every built aion extension so the
        # OS resolves them when the extension loads.
        raw = os.environ.get("AION_PY_RUNTIME_DLLS", "")
        dlls = [p for p in raw.split(os.pathsep) if p]
        if dlls:
            for ext in self.extensions:
                dest_dir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
                dest_dir.mkdir(parents=True, exist_ok=True)
                for dll in dlls:
                    src = Path(dll)
                    if src.is_file():
                        shutil.copy2(src, dest_dir / src.name)


# cffi will import the builder referenced below during the build.
setup(
    cffi_modules=["src/aion/_ffi/build.py:ffibuilder"],
    cmdclass={"build_ext": build_ext},
)
