# SPDX-License-Identifier: Apache-2.0
"""NumPy interop for `Tensor` — dtype-generic, driven by the `dtype` table.

The C ABI's `aion_tensor_create/read/write` take a `void*` + element count and a
dtype tag, so a single contiguous, correctly-aligned host buffer works for any
scalar dtype (f32/f16/i8/i32). Quantized dtypes have no host array form and raise.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .dtype import c_elem, is_quantized, normalize_dtype, numpy_dtype
from .errors import raise_for_status
from ._ffi import ffi, lib
from .enums import AionDType
from .types import ArrayLike, NDArray

if TYPE_CHECKING:
    from .context import Context
    from .tensor import Tensor


def _require_numpy():
    try:
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("NumPy is required; install aion[numpy]") from e
    return np


def _as_contiguous(array: ArrayLike, dt: AionDType):
    """Coerce `array` to a C-contiguous, writable, itemsize-aligned buffer of the
    numpy dtype matching `dt`. Returns the (possibly copied) numpy array."""
    np = _require_numpy()
    np_dt = numpy_dtype(dt)

    arr = np.asarray(array)
    if arr.dtype != np_dt:
        arr = arr.astype(np_dt, copy=False)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    if not bool(arr.flags.writeable):
        # cffi.from_buffer generally requires a writable buffer.
        arr = np.array(arr, dtype=np_dt, copy=True)
    # The C ABI enforces natural alignment for the element type.
    if (int(arr.ctypes.data) % np_dt.itemsize) != 0:
        arr = np.array(arr, dtype=np_dt, copy=True)
    return arr


def tensor_to_numpy(tensor: "Tensor") -> NDArray:
    """Copy a tensor into a new NumPy array (scalar dtypes only)."""
    np = _require_numpy()

    dt = tensor.dtype
    if is_quantized(dt):
        raise NotImplementedError(f"{dt.name}: quantized tensors have no numpy representation")

    shape = tensor.shape
    out = np.empty(shape, dtype=numpy_dtype(dt))
    n = int(out.size)

    buf = ffi.from_buffer(c_elem(dt) + "[]", out)
    st = lib.aion_tensor_read(tensor.ptr, int(dt), buf, n)
    raise_for_status(st, tensor._ctx_owner.ptr, what="aion_tensor_read")
    return out


def tensor_from_numpy(ctx: "Context", array: ArrayLike, *, dtype=None) -> "Tensor":
    """Create a tensor from a NumPy array. `dtype` defaults to the array's dtype
    (mapped to the nearest Aion scalar dtype)."""
    np = _require_numpy()
    if dtype is not None:
        dt = normalize_dtype(dtype)
    else:
        dt = normalize_dtype(np.asarray(array).dtype)
    if is_quantized(dt):
        raise NotImplementedError(f"{dt.name}: use Tensor.quantize for quantized tensors")

    arr = _as_contiguous(array, dt)

    shape = list(arr.shape)
    rank = len(shape)
    c_shape = ffi.new("size_t[]", [int(x) for x in shape]) if rank != 0 else ffi.NULL

    n = int(arr.size)
    buf = ffi.from_buffer(c_elem(dt) + "[]", arr)

    out_t = ffi.new("AionTensor**")
    st = lib.aion_tensor_create(ctx.ptr, int(dt), rank, c_shape, buf, n, out_t)
    raise_for_status(st, ctx.ptr, what="aion_tensor_create")

    from .tensor import Tensor

    return Tensor._from_handle(ctx, out_t[0], dtype=dt, shape=tuple(shape))


def tensor_write_from_numpy(tensor: "Tensor", array: ArrayLike) -> None:
    """Write into an existing tensor from a NumPy array (dtype = tensor's;
    shape must match)."""
    dt = tensor.dtype
    if is_quantized(dt):
        raise NotImplementedError(f"{dt.name}: quantized tensors have no numpy write path")

    arr = _as_contiguous(array, dt)

    expected_shape = tensor.shape
    if tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(f"shape mismatch: tensor {expected_shape} vs array {arr.shape}")

    n = int(arr.size)
    buf = ffi.from_buffer(c_elem(dt) + "[]", arr)
    st = lib.aion_tensor_write(tensor.ptr, int(dt), buf, n)
    raise_for_status(st, tensor._ctx_owner.ptr, what="aion_tensor_write")
