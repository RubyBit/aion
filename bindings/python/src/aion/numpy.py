# SPDX-License-Identifier: Apache-2.0
"""NumPy interop for `Tensor` — dtype-generic, driven by the `dtype` table.

The C ABI's `aion_tensor_create/read/write` take a `void*` + element count and a
dtype tag, so a single contiguous, correctly-aligned host buffer works for any
scalar dtype (f32/f16/i8/i32). Quantized dtypes have no host array form and raise.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .dtype import dtype_name, is_quantized, normalize_dtype, numpy_dtype
from ._ffi.runtime import (
    create_tensor_from_buffer,
    read_tensor_into_buffer,
    write_tensor_from_buffer,
)
from .enums import AionDType
from .types import ArrayLike, DTypeLike, NDArray

if TYPE_CHECKING:
    from .context import Context
    from .tensor import Tensor


def _require_numpy():
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover
        raise ImportError("NumPy is required; install aion[numpy]") from e
    return np


def _as_contiguous(array: ArrayLike, dt: AionDType):
    """Coerce `array` to a C-contiguous, writable, itemsize-aligned buffer of the
    numpy dtype matching `dt`. Returns the (possibly copied) numpy array."""
    np = _require_numpy()
    np_dt = numpy_dtype(dt)

    arr = np.asarray(cast(Any, array))
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


def _tensor_to_numpy(tensor: "Tensor") -> NDArray:
    """Copy a tensor into a new NumPy array (scalar dtypes only)."""
    np = _require_numpy()

    dt = tensor.dtype
    if is_quantized(dt):
        raise NotImplementedError(
            f"{dtype_name(dt)}: quantized tensors have no numpy representation"
        )

    shape = tensor.shape
    out = np.empty(shape, dtype=numpy_dtype(dt))
    n = int(out.size)

    read_tensor_into_buffer(
        tensor._ctx_owner.ptr, tensor.ptr, out, n
    )
    return out


def _tensor_from_numpy(
    ctx: "Context", array: ArrayLike, *, dtype: DTypeLike | None = None
) -> "Tensor":
    """Create a tensor from a NumPy array. `dtype` defaults to the array's dtype
    (mapped to the nearest Aion scalar dtype)."""
    np = _require_numpy()
    if dtype is not None:
        dt = normalize_dtype(dtype)
    else:
        dt = normalize_dtype(np.asarray(cast(Any, array)).dtype)
    if is_quantized(dt):
        raise NotImplementedError(
            f"{dtype_name(dt)}: use Tensor.quantize for quantized tensors"
        )

    arr = _as_contiguous(array, dt)
    if arr.ndim == 0:
        # Aion currently represents scalars as one-element vectors.
        arr = arr.reshape(1)

    shape = list(arr.shape)
    n = int(arr.size)
    handle = create_tensor_from_buffer(
        ctx.ptr, dt, shape, arr, n
    )

    from .tensor import Tensor

    return Tensor._from_handle(ctx, handle, dtype=dt, shape=tuple(shape))


def _copy_numpy_into_tensor(tensor: "Tensor", array: ArrayLike) -> None:
    """Write into an existing tensor from a NumPy array (dtype = tensor's;
    shape must match)."""
    np = _require_numpy()
    dt = tensor.dtype
    if is_quantized(dt):
        raise NotImplementedError(
            f"{dtype_name(dt)}: quantized tensors have no numpy write path"
        )

    arr = _as_contiguous(array, dt)

    expected_shape = tensor.shape
    if arr.ndim == 0:
        arr = np.full(expected_shape, arr.item(), dtype=numpy_dtype(dt))
    elif tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(f"shape mismatch: tensor {expected_shape} vs array {arr.shape}")

    n = int(arr.size)
    write_tensor_from_buffer(
        tensor._ctx_owner.ptr, tensor.ptr, arr, n
    )
