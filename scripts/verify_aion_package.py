#!/usr/bin/env python3
"""Verify an `.aion` package matches the expected quantization policy.

This is a byte-level file-format parser that never materializes tensor data, so it
can validate multi-GB packages in seconds. It checks:

1. Header magic/version.
2. Section directory is self-consistent.
3. Every initializer has a known encoding (plain or quantized) and one of the
   dtype sets passed via `--allow-*` flags.
4. Reports per-dtype totals so you can spot accidental f32 expansion.

Usage:
  uv run --project bindings/python scripts/verify_aion_package.py \\
      models/gemma/gemma4_e2b_q8.aion \\
      --allow-plain f16 \\
      --allow-quantized q8_0
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Reuse the writer's constants to avoid drift between reader and writer.
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _aion_writer as aw  # type: ignore  # noqa: E402


_DTYPE_NAMES: Dict[int, str] = {
    aw.DType.f32: "f32",
    aw.DType.f16: "f16",
    aw.DType.i8: "i8",
    aw.DType.q4_0: "q4_0",
    aw.DType.q8_0: "q8_0",
    aw.DType.i32: "i32",
}
_DTYPE_CODES: Dict[str, int] = {v: k for k, v in _DTYPE_NAMES.items()}


@dataclass
class _SectionRef:
    section_type: int
    flags: int
    offset: int
    size: int


@dataclass
class _InitializerRecord:
    kind: int  # 0 = plain, 1 = quantized
    plain_dtype: int
    logical_dtype: int
    scheme: str  # for quantized; "" for plain
    block_elems: int
    block_bytes: int
    quant_axis: int
    data_len: int


def _read_header(data: bytes) -> Tuple[int, int, int, int]:
    if len(data) < aw.HEADER_SIZE:
        raise ValueError("file too small for v4 header")
    if data[:4] != aw.MAGIC:
        raise ValueError(f"bad magic {data[:4]!r}")
    (version, section_count, _reserved) = struct.unpack_from("<III", data, 4)
    if version != aw.VERSION:
        raise ValueError(f"version {version} not supported (expected {aw.VERSION})")
    (dir_offset, file_size, _flags) = struct.unpack_from("<QQQ", data, 16)
    if file_size != len(data):
        raise ValueError(f"file size {len(data)} != header file_size {file_size}")
    return version, section_count, dir_offset, file_size


def _read_directory(data: bytes, dir_offset: int, section_count: int) -> List[_SectionRef]:
    refs: List[_SectionRef] = []
    cursor = dir_offset
    for _ in range(section_count):
        sec_type, flags = struct.unpack_from("<II", data, cursor)
        offset, size = struct.unpack_from("<QQ", data, cursor + 8)
        refs.append(_SectionRef(sec_type, flags, offset, size))
        cursor += aw.SECTION_DESC_SIZE
    return refs


def _read_strings_section(data: bytes, off: int, size: int) -> List[str]:
    cursor = off
    end = off + size
    count = struct.unpack_from("<I", data, cursor)[0]
    cursor += 4
    out: List[str] = []
    for _ in range(count):
        slen = struct.unpack_from("<I", data, cursor)[0]
        cursor += 4
        out.append(data[cursor : cursor + slen].decode("utf-8"))
        cursor += slen
    if cursor != end:
        raise ValueError(f"strings section: trailing bytes at {cursor} != {end}")
    return out


def _read_initializers_section(
    data: bytes, off: int, size: int, strings: List[str]
) -> List[_InitializerRecord]:
    cursor = off
    end = off + size
    count = struct.unpack_from("<I", data, cursor)[0]
    cursor += 4
    out: List[_InitializerRecord] = []
    for _ in range(count):
        (kind, plain_dt, logical_dt, _pad) = struct.unpack_from("<BBBB", data, cursor)
        cursor += 4
        (scheme_idx, block_elems, block_bytes, quant_axis, params_len) = struct.unpack_from(
            "<IIIiI", data, cursor
        )
        cursor += 5 * 4
        (data_len,) = struct.unpack_from("<Q", data, cursor)
        cursor += 8

        scheme = ""
        if kind == 1:
            if scheme_idx == aw.INVALID_INDEX_U32 or scheme_idx >= len(strings):
                raise ValueError(f"initializer: bad scheme index {scheme_idx}")
            scheme = strings[scheme_idx]

        # Skip params + data bytes; we only want metadata.
        cursor += params_len
        cursor += data_len

        out.append(
            _InitializerRecord(
                kind=kind,
                plain_dtype=plain_dt,
                logical_dtype=logical_dt,
                scheme=scheme,
                block_elems=block_elems,
                block_bytes=block_bytes,
                quant_axis=quant_axis,
                data_len=data_len,
            )
        )
    if cursor != end:
        raise ValueError(f"initializers section: trailing bytes at {cursor} != {end}")
    return out


def _parse(args: argparse.Namespace) -> int:
    with open(args.path, "rb") as fh:
        data = fh.read()

    _ver, sec_count, dir_off, _fsize = _read_header(data)
    refs = _read_directory(data, dir_off, sec_count)

    ref_by_type = {r.section_type: r for r in refs}
    if aw.SectionType.strings not in ref_by_type:
        raise ValueError("strings section missing")
    if aw.SectionType.tensors not in ref_by_type:
        raise ValueError("tensors (initializers) section missing")

    strings_ref = ref_by_type[aw.SectionType.strings]
    strings = _read_strings_section(data, strings_ref.offset, strings_ref.size)

    tensors_ref = ref_by_type[aw.SectionType.tensors]
    inits = _read_initializers_section(data, tensors_ref.offset, tensors_ref.size, strings)

    allow_plain = set(_DTYPE_CODES[name] for name in args.allow_plain)
    allow_quant = set(args.allow_quantized)

    per_dtype_bytes: Dict[str, int] = {}
    violations: List[str] = []

    for i, rec in enumerate(inits):
        if rec.kind == 0:
            if rec.plain_dtype not in allow_plain:
                violations.append(
                    f"[{i}] plain initializer dtype {_DTYPE_NAMES.get(rec.plain_dtype, rec.plain_dtype)} "
                    f"not in allow-plain {sorted(args.allow_plain)}"
                )
            key = f"plain.{_DTYPE_NAMES.get(rec.plain_dtype, '?')}"
        elif rec.kind == 1:
            if rec.scheme not in allow_quant:
                violations.append(f"[{i}] quantized scheme {rec.scheme!r} not in allow-quantized {sorted(allow_quant)}")
            key = f"quant.{rec.scheme}"
        else:
            violations.append(f"[{i}] unknown kind {rec.kind}")
            key = "unknown"
        per_dtype_bytes[key] = per_dtype_bytes.get(key, 0) + rec.data_len

    total = sum(per_dtype_bytes.values())
    print(f"file: {args.path}")
    print(f"initializers: {len(inits)}   total initializer bytes: {total/1e9:.3f} GB")
    print("per-encoding totals:")
    for k, v in sorted(per_dtype_bytes.items(), key=lambda kv: -kv[1]):
        print(f"  {k:12s}  {v/1e9:.3f} GB")

    if violations:
        print(f"\nFAIL: {len(violations)} violation(s):")
        for v in violations[:20]:
            print(f"  {v}")
        return 1

    if args.max_gb is not None and total / 1e9 > args.max_gb:
        print(f"\nFAIL: total {total/1e9:.3f} GB > budget {args.max_gb} GB")
        return 1

    print("\nOK")
    return 0


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="path to .aion file")
    ap.add_argument(
        "--allow-plain",
        nargs="*",
        default=["f16"],
        choices=sorted(_DTYPE_CODES.keys()),
        help="permitted scalar dtypes for plain initializers",
    )
    ap.add_argument(
        "--allow-quantized",
        nargs="*",
        default=["q8_0"],
        help="permitted schemes for quantized initializers",
    )
    ap.add_argument(
        "--max-gb",
        type=float,
        default=None,
        help="reject if total initializer bytes exceed this many GB",
    )
    return _parse(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
