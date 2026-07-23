# SPDX-License-Identifier: Apache-2.0
"""The deliberately untyped edge of the Python bindings.

Only modules in :mod:`aion._ffi` may import ``ffi`` and ``lib`` from here.
Application-facing modules use the typed façade and opaque handle classes
instead of manipulating CFFI cdata directly.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

if os.name == "nt":
    _pkg_dir = Path(__file__).resolve().parent.parent
    if any(_pkg_dir.glob("*.dll")):
        try:
            os.add_dll_directory(str(_pkg_dir))
        except (OSError, AttributeError):  # pragma: no cover
            pass

try:
    from .._aion_cffi import ffi as _ffi, lib as _lib
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Failed to import the compiled Aion extension module. "
        "This usually means the native extension was not built/installed for your Python."
    ) from e

# CFFI's runtime cdata classes are private and depend on a string declaration.
# Keep that dynamism explicit and contained in this module.
ffi: Any = _ffi
lib: Any = _lib

__all__ = ["ffi", "lib"]
