from __future__ import annotations

from dataclasses import dataclass

from ._ffi import ffi, lib
from .enums import AionStatus


@dataclass
class AionError(RuntimeError):
    status: AionStatus
    message: str

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.status.name}: {self.message}"


def _status_string(status: int) -> str:
    try:
        s = lib.aion_status_string(status)
    except Exception:
        return f"AionStatus({status})"
    if s == ffi.NULL:
        return f"AionStatus({status})"
    return ffi.string(s).decode("utf-8", errors="replace")


def get_last_error_message(ctx) -> str:
    """Best-effort fetch of the context last-error string."""
    if ctx is None or ctx == ffi.NULL:
        return ""

    out_len = ffi.new("size_t*")
    # First call to get length.
    lib.aion_context_last_error_message(ctx, ffi.NULL, 0, out_len)
    n = int(out_len[0])
    if n <= 0:
        return ""

    buf = ffi.new("char[]", n + 1)
    lib.aion_context_last_error_message(ctx, buf, n + 1, out_len)
    return ffi.string(buf).decode("utf-8", errors="replace")


def raise_for_status(status: int, ctx=None, *, what: str | None = None) -> None:
    st = AionStatus(int(status))
    if st == AionStatus.AION_OK:
        return

    msg = get_last_error_message(ctx)
    if not msg:
        msg = _status_string(int(status))

    if what:
        msg = f"{what}: {msg}"

    raise AionError(st, msg)
