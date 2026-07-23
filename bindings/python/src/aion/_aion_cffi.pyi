# SPDX-License-Identifier: Apache-2.0
"""Typing shim for the generated CFFI extension.

CFFI cdata types depend on runtime C declaration strings, so pretending this
module has one structurally meaningful ``CData`` type is misleading. The
strongly typed contract lives in :mod:`aion._ffi.runtime` and
:mod:`aion._ffi.authoring`; only :mod:`aion._ffi._raw` consumes these values.
"""
from typing import Any

ffi: Any
lib: Any
