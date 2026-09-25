# SPDX-License-Identifier: Apache-2.0
"""Run `tools/check_ffi_header.py` (aion.h / dlpack.h <-> aion_ffi.h drift) under pytest.

CI runs the script as its own step; running it here too means drift, or a header
the checker cannot parse, fails locally, not first on CI.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_ffi_header.py"


def _checker():
    spec = importlib.util.spec_from_file_location("check_ffi_header", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ffi_header_matches_public_headers(capsys):
    rc = _checker().main()
    assert rc == 0, capsys.readouterr().out


def test_a_dlpack_mirror_that_drifts_is_reported():
    dlpack = (
        "#ifdef __cplusplus\n"
        'extern "C" {\n'
        "#endif\n"
        "typedef struct { void* data; int64_t* shape; } DLTensor;\n"
        "#ifdef __cplusplus\n"
        "}\n"
        "#endif\n"
    )
    real = '#include "dlpack/dlpack.h"\nAION_API int aion_ping(const DLTensor* t);\n'
    same = "typedef struct { void* data; int64_t* shape; } DLTensor;\nint aion_ping(const DLTensor* t);\n"
    drifted = "typedef struct { void* data; int32_t* shape; } DLTensor;\nint aion_ping(const DLTensor* t);\n"

    checker = _checker()
    assert checker._compare(real, same, dlpack) == []
    assert checker._compare(real, drifted, dlpack)[0] == "CHANGED in aion_ffi.h (type): DLTensor"
