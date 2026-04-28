# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext


def _import_build_zig():
    # setup.py lives at: bindings/python/setup.py
    python_root = Path(__file__).resolve().parent
    tools_dir = python_root / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import build_zig  # type: ignore
    finally:
        try:
            sys.path.remove(str(tools_dir))
        except ValueError:
            pass
    return build_zig


class build_ext(_build_ext):
    """Build the cffi extension, ensuring Aion is built first.

    On Windows, we build Aion as a DLL and bundle it into the wheel next to the
    extension module.
    """

    def run(self):
        _import_build_zig()

        # Use a deterministic prefix under build_temp so repeated builds reuse artifacts.
        prefix = (Path(self.build_temp).resolve() / "aion-zig-prefix")
        prefix.mkdir(parents=True, exist_ok=True)

        # Tell the cffi builder to reuse this prefix.
        os.environ["AION_PY_BUILD_PREFIX"] = str(prefix)

        # Keep platform defaults explicit.
        os.environ.setdefault("AION_PY_LINKAGE", "static")

        # Default to native CPU tuning for local builds unless explicitly overridden.
        os.environ.setdefault("AION_PY_CPU", "native")

        #artifacts = build_zig.build_aion(prefix=prefix)

        super().run()

        # If we ever build a shared library on Windows again, we can bundle it
        # here. For now we aim to static-link Aion into the extension module.

# cffi will import the builder referenced below during the build.
setup(
    cffi_modules=["src/aion/_ffi/build.py:ffibuilder"],
    cmdclass={"build_ext": build_ext},
)
