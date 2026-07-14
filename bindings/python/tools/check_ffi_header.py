# SPDX-License-Identifier: Apache-2.0
"""Drift check: ``include/aion.h`` vs the CFFI declaration header.

``aion_ffi.h`` is a hand-maintained, macro-free view of the public C API.  This
script parses both headers and compares canonical declarations for:

* every ``aion_*`` function;
* every ``Aion*`` typedef, including enums, opaque handles, and struct layouts.

The generated CFFI extension provides a second line of defence: API-mode CFFI
compiles these declarations against the real header.  This check additionally
catches declarations that were added to one header but omitted from the other.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from pycparser import c_ast, c_generator, c_parser


# pycparser parses C translation units rather than CFFI's extended dialect, so
# give it the standard integer names that CFFI supplies implicitly to cdef().
_STANDARD_TYPES = """
typedef unsigned char uint8_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef unsigned long size_t;
"""


def _prepare(text: str) -> str:
    """Reduce a public header to portable C declarations for pycparser."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    text = text.replace('extern "C" {', " ").replace("AION_API", " ")
    text = re.sub(r"^\s*#[^\n]*$", " ", text, flags=re.M)

    # Removing the __cplusplus directives leaves the extern-C closing brace at
    # the end of the real header.  The CFFI header has no such wrapper.
    text = re.sub(r"\n\s*}\s*\Z", "\n", text)
    return _STANDARD_TYPES + text


def _declarations(text: str, filename: str = "<header>") -> dict[tuple[str, str], str]:
    """Return canonical public declarations keyed by declaration kind/name."""
    tree = c_parser.CParser().parse(_prepare(text), filename=filename)
    generator = c_generator.CGenerator()
    declarations: dict[tuple[str, str], str] = {}

    for node in tree.ext:
        key: tuple[str, str] | None = None
        if isinstance(node, c_ast.Typedef) and node.name.startswith("Aion"):
            key = ("type", node.name)
        elif (
            isinstance(node, c_ast.Decl)
            and isinstance(node.type, c_ast.FuncDecl)
            and node.name is not None
            and node.name.startswith("aion_")
        ):
            key = ("function", node.name)

        if key is not None:
            if key in declarations:
                raise ValueError(f"duplicate {key[0]} declaration: {key[1]}")
            declarations[key] = " ".join(generator.visit(node).split())

    return declarations


def _compare(real: str, ffi: str) -> list[str]:
    """Return human-readable declaration differences."""
    actual = _declarations(real, "include/aion.h")
    mirror = _declarations(ffi, "aion_ffi.h")
    differences: list[str] = []

    for key in sorted(actual.keys() | mirror.keys()):
        kind, name = key
        if key not in mirror:
            differences.append(f"MISSING in aion_ffi.h ({kind}): {name}")
        elif key not in actual:
            differences.append(f"EXTRA in aion_ffi.h ({kind}): {name}")
        elif actual[key] != mirror[key]:
            differences.extend(
                (
                    f"CHANGED in aion_ffi.h ({kind}): {name}",
                    f"  aion.h:     {actual[key]}",
                    f"  aion_ffi.h: {mirror[key]}",
                )
            )

    return differences


def main() -> int:
    repo = Path(__file__).resolve().parents[3]
    real_path = repo / "include" / "aion.h"
    ffi_path = repo / "bindings" / "python" / "src" / "aion" / "_ffi" / "aion_ffi.h"
    real = real_path.read_text(encoding="utf-8")
    ffi = ffi_path.read_text(encoding="utf-8")

    try:
        differences = _compare(real, ffi)
        declarations = _declarations(real, str(real_path))
    except (c_parser.ParseError, ValueError) as exc:
        print(f"header drift check could not parse declarations: {exc}")
        return 1

    for difference in differences:
        print(difference)

    function_count = sum(kind == "function" for kind, _ in declarations)
    type_count = sum(kind == "type" for kind, _ in declarations)
    if function_count < 10 or type_count < 5:
        print(
            "extraction suspiciously small "
            f"(functions={function_count}, types={type_count}) -- script broken?"
        )
        differences.append("suspicious declaration count")

    status = "DRIFT" if differences else "OK"
    print(f"{status}: {function_count} functions, {type_count} API types compared")
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main())
