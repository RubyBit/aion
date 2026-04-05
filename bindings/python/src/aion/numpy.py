from __future__ import annotations

from typing import Any

from .errors import raise_for_status
from ._ffi import ffi, lib
from .enums import AionDType
from .types import ArrayLike


def _require_numpy():
    try:
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("NumPy is required; install aion[numpy]") from e
    return np


def _as_f32_contiguous(array: ArrayLike):
    np = _require_numpy()

    arr = np.asarray(array)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)

    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    if not bool(arr.flags.writeable):
        # cffi.from_buffer generally requires a writable buffer.
        arr = np.array(arr, dtype=np.float32, copy=True)

    if (int(arr.ctypes.data) & 3) != 0:
        # C ABI enforces f32 alignment.
        arr = np.array(arr, dtype=np.float32, copy=True)

    return arr


def tensor_to_numpy(tensor) -> Any:
    """Copy a tensor into a new NumPy array (f32 only)."""

    np = _require_numpy()

    if tensor.dtype() != AionDType.AION_DTYPE_F32:
        raise NotImplementedError("only f32 tensor I/O is supported by the current C ABI")

    shape = tensor.shape()
    out = np.empty(shape, dtype=np.float32)
    n = int(out.size)

    buf = ffi.from_buffer("float[]", out)
    st = lib.aion_tensor_read(tensor.ptr, int(AionDType.AION_DTYPE_F32), buf, n)
    raise_for_status(st, tensor._ctx_owner.ptr, what="aion_tensor_read")
    return out


def tensor_from_numpy(ctx, array: ArrayLike):
    """Create a tensor from a NumPy array (f32 only)."""

    arr = _as_f32_contiguous(array)

    shape = list(arr.shape)
    rank = len(shape)
    c_shape = ffi.new("size_t[]", [int(x) for x in shape]) if rank != 0 else ffi.NULL

    n = int(arr.size)
    buf = ffi.from_buffer("float[]", arr)

    out_t = ffi.new("AionTensor**")
    st = lib.aion_tensor_create(
        ctx.ptr,
        int(AionDType.AION_DTYPE_F32),
        rank,
        c_shape,
        buf,
        n,
        out_t,
    )
    raise_for_status(st, ctx.ptr, what="aion_tensor_create")

    from .tensor import Tensor

    return Tensor._from_handle(ctx, out_t[0], dtype=AionDType.AION_DTYPE_F32, shape=tuple(shape))


def tensor_write_from_numpy(tensor, array: ArrayLike) -> None:
    """Write into an existing tensor from a NumPy array (f32 only)."""

    arr = _as_f32_contiguous(array)

    if tensor.dtype() != AionDType.AION_DTYPE_F32:
        raise NotImplementedError("only f32 tensor I/O is supported by the current C ABI")

    expected_shape = tensor.shape()
    if tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(f"shape mismatch: tensor {expected_shape} vs array {arr.shape}")

    n = int(arr.size)
    buf = ffi.from_buffer("float[]", arr)
    st = lib.aion_tensor_write(tensor.ptr, int(AionDType.AION_DTYPE_F32), buf, n)
    raise_for_status(st, tensor._ctx_owner.ptr, what="aion_tensor_write")
