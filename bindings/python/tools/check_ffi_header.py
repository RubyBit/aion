# SPDX-License-Identifier: Apache-2.0
"""Drift check: ``include/aion.h`` vs the CFFI declaration header.

``aion_ffi.h`` is a hand-maintained, macro-free view of the public C API.  This
script parses the headers and compares canonical declarations for:

* every ``aion_*`` function and ``Aion*`` typedef (enums, opaque handles, struct
  layouts) against ``include/aion.h``;
* every ``DL*`` typedef the CFFI header mirrors against the vendored
  ``include/dlpack/dlpack.h``, which ``aion.h`` includes.

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
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
typedef unsigned long size_t;
"""

_CONDITIONAL = re.compile(r"\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)")


def _c_branches(text: str) -> str:
    """Keep what a C (not C++) compiler sees of `#if*` blocks on ``__cplusplus``.

    Every other conditional keeps both branches: in these headers they only hold
    `#define`s, which are dropped afterwards anyway.
    """
    kept: list[str] = []
    stack: list[tuple[bool, bool]] = []  # (decides on __cplusplus, branch kept)
    for line in text.splitlines():
        m = _CONDITIONAL.match(line)
        if m is None:
            if all(keep for _, keep in stack):
                kept.append(line)
            continue
        kind, rest = m.group(1), m.group(2).strip()
        if kind in ("ifdef", "ifndef", "if"):
            on_cpp = rest in ("__cplusplus", "defined(__cplusplus)")
            stack.append((on_cpp, kind == "ifndef" if on_cpp else True))
        elif kind in ("else", "elif"):
            on_cpp, keep = stack[-1]
            stack[-1] = (on_cpp, not keep if on_cpp else True)
        else:
            stack.pop()
    return "\n".join(kept)


def _prepare(text: str) -> str:
    """Reduce a header to portable C declarations for pycparser."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    text = _c_branches(text)
    text = re.sub(r"^\s*#[^\n]*$", " ", text, flags=re.M)
    text = text.replace("AION_API", " ")
    return _STANDARD_TYPES + text


def _declarations(text: str, filename: str = "<header>") -> dict[tuple[str, str], str]:
    """Return canonical declarations of `Aion*`/`DL*` types and `aion_*` functions."""
    tree = c_parser.CParser().parse(_prepare(text), filename=filename)
    generator = c_generator.CGenerator()
    declarations: dict[tuple[str, str], str] = {}

    for node in tree.ext:
        key: tuple[str, str] | None = None
        if isinstance(node, c_ast.Typedef) and node.name.startswith(("Aion", "DL")):
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


def _is_dlpack(key: tuple[str, str]) -> bool:
    return key[1].startswith("DL")


def _compare(real: str, ffi: str, dlpack: str) -> list[str]:
    """Return human-readable declaration differences."""
    # aion.h includes dlpack.h; parse them as the compiler sees them, together.
    actual = _declarations(dlpack + "\n" + real, "include/aion.h")
    mirror = _declarations(ffi, "aion_ffi.h")
    differences: list[str] = []

    for key in sorted(actual.keys() | mirror.keys()):
        kind, name = key
        if _is_dlpack(key) and key not in mirror:
            continue  # the binding mirrors only the DLPack types it uses
        if key not in mirror:
            differences.append(f"MISSING in aion_ffi.h ({kind}): {name}")
        elif key not in actual:
            source = "dlpack.h" if _is_dlpack(key) else "aion.h"
            differences.append(f"EXTRA in aion_ffi.h ({kind}), not in {source}: {name}")
        elif actual[key] != mirror[key]:
            source = "dlpack.h" if _is_dlpack(key) else "aion.h"
            differences.extend(
                (
                    f"CHANGED in aion_ffi.h ({kind}): {name}",
                    f"  {source + ':':<11} {actual[key]}",
                    f"  aion_ffi.h: {mirror[key]}",
                )
            )

    return differences


def main() -> int:
    repo = Path(__file__).resolve().parents[3]
    real_path = repo / "include" / "aion.h"
    dlpack_path = repo / "include" / "dlpack" / "dlpack.h"
    ffi_path = repo / "bindings" / "python" / "src" / "aion" / "_ffi" / "aion_ffi.h"
    real = real_path.read_text(encoding="utf-8")
    dlpack = dlpack_path.read_text(encoding="utf-8")
    ffi = ffi_path.read_text(encoding="utf-8")

    try:
        differences = _compare(real, ffi, dlpack)
        declarations = _declarations(ffi, str(ffi_path))
    except (c_parser.ParseError, ValueError) as exc:
        print(f"header drift check could not parse declarations: {exc}")
        return 1

    for difference in differences:
        print(difference)

    function_count = sum(kind == "function" for kind, _ in declarations)
    type_count = sum(kind == "type" and not name.startswith("DL") for kind, name in declarations)
    dlpack_count = sum(name.startswith("DL") for _, name in declarations)
    if function_count < 10 or type_count < 5 or dlpack_count < 3:
        print(
            "extraction suspiciously small "
            f"(functions={function_count}, types={type_count}, dlpack={dlpack_count}) -- script broken?"
        )
        differences.append("suspicious declaration count")

    status = "DRIFT" if differences else "OK"
    print(f"{status}: {function_count} functions, {type_count} API types, {dlpack_count} DLPack types compared")
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main())
