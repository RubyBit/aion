# SPDX-License-Identifier: Apache-2.0
"""Translate native status codes into the public Python exception."""
from __future__ import annotations

from ..enums import AionStatus
from ..errors import AionError
from ._raw import ffi, lib
from .handles import ContextHandle


def _status_string(status: int) -> str:
    try:
        value = lib.aion_status_string(status)
    except Exception:
        return f"AionStatus({status})"
    if value == ffi.NULL:
        return f"AionStatus({status})"
    return ffi.string(value).decode("utf-8", errors="replace")


def _last_error_message(ctx: ContextHandle | None) -> str:
    if ctx is None:
        return ""
    raw_ctx = ctx.raw
    out_len = ffi.new("size_t*")
    lib.aion_context_last_error_message(raw_ctx, ffi.NULL, 0, out_len)
    size = int(out_len[0])
    if size <= 0:
        return ""
    buf = ffi.new("char[]", size + 1)
    lib.aion_context_last_error_message(raw_ctx, buf, size + 1, out_len)
    return ffi.string(buf).decode("utf-8", errors="replace")


def raise_for_status(
    status: int,
    ctx: ContextHandle | None = None,
    *,
    what: str | None = None,
) -> None:
    typed_status = AionStatus(int(status))
    if typed_status == AionStatus.AION_OK:
        return
    message = _last_error_message(ctx) or _status_string(int(status))
    if what:
        message = f"{what}: {message}"
    raise AionError(typed_status, message)


__all__ = ["raise_for_status"]
