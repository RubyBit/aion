# SPDX-License-Identifier: Apache-2.0
"""ABI drift guard: `aion.enums` must match the compiled cffi constants.

The enum values are hand-maintained in three places (`include/aion.h`,
`src/aion/_ffi/aion_ffi.h`, `src/aion/enums.py`). cffi compiles the ffi header,
so `lib` carries the header's ground-truth values; every python enum member
must exist there with the same value. `tools/check_ffi_header.py` covers the
aion.h <-> aion_ffi.h leg.
"""
from __future__ import annotations

import enum

import aion.enums as aion_enums
from aion._ffi._raw import lib


def _enum_classes():
    for name in dir(aion_enums):
        obj = getattr(aion_enums, name)
        if isinstance(obj, type) and issubclass(obj, enum.Enum) and obj is not enum.Enum:
            yield obj


def test_python_enums_match_compiled_abi():
    checked = 0
    mismatches = []
    for cls in _enum_classes():
        for member in cls:
            if not hasattr(lib, member.name):
                mismatches.append(f"{cls.__name__}.{member.name}: missing from ffi header")
                continue
            lib_val = int(getattr(lib, member.name))
            if lib_val != int(member.value):
                mismatches.append(
                    f"{cls.__name__}.{member.name}: enums.py={int(member.value)} ffi={lib_val}")
            checked += 1
    assert not mismatches, "enum drift vs aion_ffi.h:\n  " + "\n  ".join(mismatches)
    assert checked >= 10, f"suspiciously few enum members checked ({checked})"
