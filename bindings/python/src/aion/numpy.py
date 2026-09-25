# SPDX-License-Identifier: Apache-2.0
"""NumPy interop for `Tensor` — dtype-generic, driven by the `dtype` table.

Arrays cross the C ABI as DLPack views (`_ffi.dlpack`), so any strided array is
read or written in place; only a dtype the tensor cannot take directly is
converted first. Quantized dtypes have no host array form and raise.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .dtype import dtype_name, is_quantized, normalize_dtype, numpy_dtype
from ._ffi.dlpack import view_of
from ._ffi.runtime import create_empty_tensor, read_view, write_view
from .enums import AionDType
from .types import ArrayLike, DTypeLike, NDArray

if TYPE_CHECKING:
    from .context import Context
    from .tensor import Tensor


def _require_numpy():
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover
        raise ImportError("NumPy is required; install aion-engine[numpy]") from e
    return np


def _as_array(array: ArrayLike, dt: AionDType):
    """`array` as a numpy array of the numpy dtype matching `dt` (a copy only when
    the dtype differs; strides are kept)."""
    np = _require_numpy()
    arr = np.asarray(cast(Any, array))
    np_dt = numpy_dtype(dt)
    return arr if arr.dtype == np_dt else arr.astype(np_dt)


def _tensor_to_numpy(tensor: "Tensor") -> NDArray:
    """Copy a tensor into a new NumPy array (scalar dtypes only)."""
    np = _require_numpy()

    dt = tensor.dtype
    if is_quantized(dt):
        raise NotImplementedError(
            f"{dtype_name(dt)}: quantized tensors have no numpy representation"
        )

    out = np.empty(tensor.shape, dtype=numpy_dtype(dt))
    read_view(tensor._ctx_owner.ptr, tensor.ptr, view_of(out))
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

    arr = _as_array(array, dt)
    if arr.ndim == 0:
        # Aion currently represents scalars as one-element vectors.
        arr = arr.reshape(1)

    shape = tuple(int(d) for d in arr.shape)
    handle = create_empty_tensor(ctx.ptr, dt, shape)

    from .tensor import Tensor

    t = Tensor._from_handle(ctx, handle, dtype=dt, shape=shape)
    write_view(ctx.ptr, t.ptr, view_of(arr))
    return t


def _copy_numpy_into_tensor(tensor: "Tensor", array: ArrayLike) -> None:
    """Write into an existing tensor from a NumPy array (dtype = tensor's;
    shape must match)."""
    np = _require_numpy()
    dt = tensor.dtype
    if is_quantized(dt):
        raise NotImplementedError(
            f"{dtype_name(dt)}: quantized tensors have no numpy write path"
        )

    arr = _as_array(array, dt)
    expected_shape = tensor.shape
    if arr.ndim == 0:
        arr = np.full(expected_shape, arr.item(), dtype=numpy_dtype(dt))
    elif tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(f"shape mismatch: tensor {expected_shape} vs array {arr.shape}")

    write_view(tensor._ctx_owner.ptr, tensor.ptr, view_of(arr))
