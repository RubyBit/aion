# SPDX-License-Identifier: Apache-2.0
"""INTERNAL model-building helpers. No stability guarantees.

`aion._writer.format` encodes the on-disk `.aion` container (source of truth:
`src/aion/storage/aion_file/*.zig`); `aion._writer.builder.Builder` is the
ergonomic graph-construction layer used by the converter scripts and the test
fixtures. Slated for retirement once the core-lib Zig Builder is exposed
through the C ABI.

Not imported by the `aion` runtime package itself — importing this module pulls
in numpy.
"""
from . import format
from .builder import Builder

__all__ = ["Builder", "format"]
