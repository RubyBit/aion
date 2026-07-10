# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from typing import Iterable, Sequence, overload

from .device import DeviceLike, _device_to_str, _normalize_device
from .errors import raise_for_status
from ._ffi import ffi, lib
from .enums import AionDType
from .types import ArrayLike, F32ArrayLike, NDArrayF32


def _as_shape(shape: Sequence[int]) -> list[int]:
    if isinstance(shape, (list, tuple)):
        shp = [int(x) for x in shape]
    else:
        shp = [int(x) for x in shape]
    if any(d < 0 for d in shp):
        raise ValueError("shape dims must be >= 0")
    return shp


def _elem_count(shape: Sequence[int]) -> int:
    if len(shape) == 0:
        return 1
    return int(math.prod(int(x) for x in shape))


def _flatten_nested(data) -> tuple[tuple[int, ...], list[float]]:
    """Infer shape and flatten nested Python sequences to f32 values."""

    if isinstance(data, (int, float)):
        return (), [float(data)]

    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return (0,), []

        child_shape, child_vals = _flatten_nested(data[0])
        out_vals: list[float] = list(child_vals)
        for item in data[1:]:
            shp, vals = _flatten_nested(item)
            if shp != child_shape:
                raise ValueError(f"ragged nested sequence is not supported: {shp} != {child_shape}")
            out_vals.extend(vals)

        return (len(data),) + child_shape, out_vals

    # Treat other iterables (e.g. range) as 1D.
    try:
        seq = list(data)
    except TypeError as e:
        raise TypeError("Tensor data must be a scalar or nested sequence") from e
    return _flatten_nested(seq)


def _flatten_nested_i32(data) -> tuple[tuple[int, ...], list[int]]:
    """Infer shape and flatten nested Python sequences to i32 values."""

    if isinstance(data, bool):
        return (), [int(data)]
    if isinstance(data, int):
        return (), [int(data)]

    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return (0,), []

        child_shape, child_vals = _flatten_nested_i32(data[0])
        out_vals: list[int] = list(child_vals)
        for item in data[1:]:
            shp, vals = _flatten_nested_i32(item)
            if shp != child_shape:
                raise ValueError(f"ragged nested sequence is not supported: {shp} != {child_shape}")
            out_vals.extend(vals)

        return (len(data),) + child_shape, out_vals

    # Treat other iterables (e.g. range) as 1D.
    try:
        seq = list(data)
    except TypeError as e:
        raise TypeError("Tensor data must be a scalar or nested sequence") from e
    return _flatten_nested_i32(seq)


class Tensor:
    """Owns an `AionTensor*` handle; underlying storage is owned by the Context."""

    def __init__(
        self,
        data,
        *,
        ctx=None,
        dtype: AionDType | None = None,
        device: DeviceLike = None,
    ):
        """Create a tensor from Python data.

        Examples:
            Tensor([1, 2, 3])
            Tensor([[1, 2], [3, 4]], dtype=AionDType.AION_DTYPE_F32)
            Tensor([1, 2, 3], device="gpu")   # built on CPU, then migrated

        `device` migrates the tensor after it is built and written on the CPU
        (move semantics: the host copy is freed). Host `read_*`/`write_*` then
        fail until you migrate back with ``.to("cpu")``.
        """

        from .context import get_default_context

        if ctx is None:
            ctx = get_default_context()
        if dtype is None:
            dtype = AionDType.AION_DTYPE_F32
        if dtype in (AionDType.AION_DTYPE_Q4_0, AionDType.AION_DTYPE_Q8_0):
            raise NotImplementedError("quantized tensor construction is not supported by the current C ABI")

        # NumPy fast-path when available.
        try:
            import numpy as np  # type: ignore
        except ImportError:
            np = None  # type: ignore

        if dtype == AionDType.AION_DTYPE_F32:
            if np is not None and isinstance(data, np.ndarray):
                arr = np.asarray(data, dtype=np.float32)
                shape = tuple(int(x) for x in arr.shape)
                vals = [float(v) for v in arr.reshape(-1)]
            else:
                shape, vals = _flatten_nested(data)
        elif dtype == AionDType.AION_DTYPE_I32:
            if np is not None and isinstance(data, np.ndarray):
                arr = np.asarray(data, dtype=np.int32)
                shape = tuple(int(x) for x in arr.shape)
                vals = [int(v) for v in arr.reshape(-1)]
            else:
                shape, vals = _flatten_nested_i32(data)
        else:
            raise NotImplementedError(f"tensor construction not implemented for dtype={dtype}")

        shp = _as_shape(shape)
        n = _elem_count(shp)
        if len(vals) != n:
            raise ValueError(f"expected {n} values for shape {shp}, got {len(vals)}")

        # Aion currently represents scalars as shape (1,), not rank-0 tensors.
        if len(shp) == 0:
            if len(vals) != 1:
                raise ValueError(f"expected 1 value for scalar tensor, got {len(vals)}")
            if dtype == AionDType.AION_DTYPE_F32:
                t = self.scalar_f32(ctx, float(vals[0]))
            elif dtype == AionDType.AION_DTYPE_I32:
                t = self.scalar_i32(ctx, int(vals[0]))
            else:
                raise NotImplementedError(f"scalar tensor construction not implemented for dtype={dtype}")
            self._adopt(t)
        else:
            rank = len(shp)
            c_shape = ffi.new("size_t[]", [int(x) for x in shp]) if rank != 0 else ffi.NULL

            if dtype == AionDType.AION_DTYPE_F32:
                c_vals = ffi.new("float[]", [float(v) for v in vals])
            elif dtype == AionDType.AION_DTYPE_I32:
                c_vals = ffi.new("int32_t[]", [int(v) for v in vals])
            else:
                raise NotImplementedError(f"tensor construction not implemented for dtype={dtype}")

            out_t = ffi.new("AionTensor**")
            st = lib.aion_tensor_create(
                ctx.ptr,
                int(dtype),
                rank,
                c_shape,
                c_vals,
                n,
                out_t,
            )
            raise_for_status(st, ctx.ptr, what="aion_tensor_create")
            self._init_from_handle(ctx, out_t[0], dtype=dtype, shape=tuple(shp))

        if device is not None:
            self.to(device)

    def _init_from_handle(self, ctx, ptr, *, dtype: AionDType | None = None, shape: tuple[int, ...] | None = None) -> None:
        self._ctx_owner = ctx
        self._ctx_owner._register_child(self)
        self._t = ptr
        self._closed = False
        self._dtype_cache = dtype
        self._shape_cache = shape

    def _adopt(self, other: "Tensor") -> None:
        self._init_from_handle(other._ctx_owner, other._t, dtype=other._dtype_cache, shape=other._shape_cache)
        other._t = ffi.NULL
        other._closed = True

    @classmethod
    def _from_handle(
        cls,
        ctx,
        ptr,
        *,
        dtype: AionDType | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> "Tensor":
        self = cls.__new__(cls)
        self._init_from_handle(ctx, ptr, dtype=dtype, shape=shape)
        return self

    @property
    def ptr(self):
        return self._t

    def to(self, device: DeviceLike) -> "Tensor":
        """Migrate this tensor to `device` (move semantics; returns self).

        The source-device copy is freed. Migrating off the CPU makes host
        `read_*`/`write_*` fail until you migrate back with ``.to("cpu")``.
        Idempotent when already on the target device. Accepts ``"cpu"``,
        ``"gpu"``, ``"gpu:N"``, ``(kind, index)``, or an `AionDeviceKind`.
        """

        kind, index = _normalize_device(device)
        st = lib.aion_tensor_to(self._t, int(kind), int(index))
        raise_for_status(st, self._ctx_owner.ptr, what="aion_tensor_to")
        return self

    def device(self) -> str:
        """The device this tensor is resident on: ``"cpu"`` or ``"gpu:N"``."""

        out_kind = ffi.new("AionDeviceKind*")
        out_index = ffi.new("uint32_t*")
        st = lib.aion_tensor_device(self._t, out_kind, out_index)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_tensor_device")
        return _device_to_str(int(out_kind[0]), int(out_index[0]))

    def close(self) -> None:
        if self._closed:
            return
        lib.aion_tensor_destroy(self._t)
        try:
            self._ctx_owner._unregister_child(self)
        except Exception:
            pass
        self._t = ffi.NULL
        self._closed = True

    def __enter__(self) -> "Tensor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):  # pragma: no cover
        try:
            if not getattr(self, "_closed", True):
                lib.aion_tensor_destroy(getattr(self, "_t", ffi.NULL))
        except Exception:
            pass

    @classmethod
    def empty(
        cls,
        ctx,
        shape: Sequence[int],
        *,
        dtype: AionDType | None = None,
        device: DeviceLike = None,
    ) -> "Tensor":
        shp = _as_shape(shape)
        if dtype is None:
            dtype = AionDType.AION_DTYPE_F32

        rank = len(shp)
        c_shape = ffi.new("size_t[]", [int(x) for x in shp]) if rank != 0 else ffi.NULL
        out_t = ffi.new("AionTensor**")
        st = lib.aion_tensor_create_empty(ctx.ptr, int(dtype), rank, c_shape, out_t)
        raise_for_status(st, ctx.ptr, what="aion_tensor_create_empty")
        t = cls._from_handle(ctx, out_t[0], dtype=dtype, shape=tuple(shp))
        if device is not None:
            t.to(device)
        return t

    @classmethod
    def empty_tiled(
        cls,
        ctx,
        shape: Sequence[int],
        tile_shape: Sequence[int],
        *,
        dtype: AionDType | None = None,
    ) -> "Tensor":
        """Create an empty tensor with an explicit per-axis tile shape.

        Use this when you want to pin a specific tile layout — e.g. to skip the
        one-time retile copy the compiler would otherwise insert when an op needs
        a different tiling. For most cases `Tensor.empty` is sufficient: the graph
        compiler retiles inputs as needed (e.g. KV caches consumed by
        `KVCacheAppend` are coerced to head-dim-contiguous on first run).
        """
        shp = _as_shape(shape)
        tshp = _as_shape(tile_shape)
        if len(tshp) != len(shp):
            raise ValueError(f"tile_shape rank {len(tshp)} != shape rank {len(shp)}")
        if dtype is None:
            dtype = AionDType.AION_DTYPE_F32

        rank = len(shp)
        c_shape = ffi.new("size_t[]", [int(x) for x in shp]) if rank != 0 else ffi.NULL
        c_tile = ffi.new("size_t[]", [int(x) for x in tshp]) if rank != 0 else ffi.NULL
        out_t = ffi.new("AionTensor**")
        st = lib.aion_tensor_create_empty_tiled(ctx.ptr, int(dtype), rank, c_shape, c_tile, out_t)
        raise_for_status(st, ctx.ptr, what="aion_tensor_create_empty_tiled")
        return cls._from_handle(ctx, out_t[0], dtype=dtype, shape=tuple(shp))

    @classmethod
    def zeros(
        cls,
        shape: Sequence[int],
        *,
        ctx=None,
        dtype: AionDType | None = None,
        device: DeviceLike = None,
    ) -> "Tensor":
        """Create a zero-initialized tensor.

        If `ctx` is omitted, the process-wide default context is used. When
        `device` is set, the tensor is zeroed on the CPU and then migrated.
        """

        from .context import get_default_context

        if ctx is None:
            ctx = get_default_context()
        t = cls.empty(ctx, shape, dtype=dtype)
        t.zero()
        if device is not None:
            t.to(device)
        return t

    @classmethod
    def from_f32(
        cls,
        ctx,
        shape: Sequence[int],
        values: Iterable[float],
        *,
        device: DeviceLike = None,
    ) -> "Tensor":
        shp = _as_shape(shape)
        n = _elem_count(shp)
        # Fast-path: NumPy array-like.
        try:
            import numpy as np  # type: ignore
        except ImportError:
            np = None  # type: ignore

        if np is not None and isinstance(values, np.ndarray):
            from .numpy import tensor_from_numpy

            arr = np.asarray(values, dtype=np.float32)
            if tuple(arr.shape) != tuple(shp):
                raise ValueError(f"expected values with shape {tuple(shp)}, got {arr.shape}")
            t = tensor_from_numpy(ctx, arr)
            if device is not None:
                t.to(device)
            return t

        vals = list(values)
        if len(vals) != n:
            raise ValueError(f"expected {n} values for shape {shp}, got {len(vals)}")

        # Aion currently represents scalars as shape (1,), not rank-0 tensors.
        # Keep Python ergonomic by accepting shape=() as a scalar convenience.
        if len(shp) == 0:
            if len(vals) != 1:
                raise ValueError(f"expected 1 value for scalar tensor, got {len(vals)}")
            t = cls.scalar_f32(ctx, float(vals[0]))
            if device is not None:
                t.to(device)
            return t

        rank = len(shp)
        c_shape = ffi.new("size_t[]", [int(x) for x in shp]) if rank != 0 else ffi.NULL
        c_vals = ffi.new("float[]", [float(v) for v in vals])
        out_t = ffi.new("AionTensor**")
        st = lib.aion_tensor_create(
            ctx.ptr,
            int(AionDType.AION_DTYPE_F32),
            rank,
            c_shape,
            c_vals,
            n,
            out_t,
        )
        raise_for_status(st, ctx.ptr, what="aion_tensor_create")
        t = cls._from_handle(ctx, out_t[0], dtype=AionDType.AION_DTYPE_F32, shape=tuple(shp))
        if device is not None:
            t.to(device)
        return t

    @classmethod
    def scalar_f32(cls, ctx, value: float) -> "Tensor":
        # Scalars are encoded as shape (1,) in the current Aion runtime.
        t = cls.empty(ctx, (1,), dtype=AionDType.AION_DTYPE_F32)
        t.write_f32([float(value)])
        return t

    @classmethod
    def scalar_i32(cls, ctx, value: int) -> "Tensor":
        t = cls.empty(ctx, (1,), dtype=AionDType.AION_DTYPE_I32)
        t.write_i32([int(value)])
        return t

    def numel(self) -> int:
        return _elem_count(self.shape())

    def dtype(self) -> AionDType:
        if self._dtype_cache is None:
            self._dtype_cache = AionDType(int(lib.aion_tensor_dtype(self._t)))
        return self._dtype_cache

    def shape(self) -> tuple[int, ...]:
        if self._shape_cache is not None:
            return self._shape_cache

        rank = int(lib.aion_tensor_rank(self._t))
        if rank == 0:
            self._shape_cache = ()
            return self._shape_cache

        dims = ffi.new("size_t[]", rank)
        st = lib.aion_tensor_shape(self._t, dims, rank)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_tensor_shape")
        self._shape_cache = tuple(int(dims[i]) for i in range(rank))
        return self._shape_cache

    def read_f32(self) -> list[float]:
        if self.dtype() != AionDType.AION_DTYPE_F32:
            raise NotImplementedError("dtype mismatch: expected f32")

        shp = self.shape()
        n = _elem_count(shp)
        buf = ffi.new("float[]", n)
        st = lib.aion_tensor_read(self._t, int(AionDType.AION_DTYPE_F32), buf, n)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_tensor_read")
        return [float(buf[i]) for i in range(n)]

    def read_i32(self) -> list[int]:
        if self.dtype() != AionDType.AION_DTYPE_I32:
            raise NotImplementedError("dtype mismatch: expected i32")

        shp = self.shape()
        n = _elem_count(shp)
        buf = ffi.new("int32_t[]", n)
        st = lib.aion_tensor_read(self._t, int(AionDType.AION_DTYPE_I32), buf, n)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_tensor_read")
        return [int(buf[i]) for i in range(n)]

    @overload
    def write_f32(self, values: Iterable[float]) -> None: ...

    @overload
    def write_f32(self, values: F32ArrayLike) -> None: ...

    def write_f32(self, values) -> None:
        if self.dtype() != AionDType.AION_DTYPE_F32:
            raise NotImplementedError("dtype mismatch: expected f32")

        # Fast-path: NumPy array-like.
        try:
            import numpy as np  # type: ignore
        except ImportError:
            np = None  # type: ignore

        if np is not None and isinstance(values, np.ndarray):
            from .numpy import tensor_write_from_numpy

            tensor_write_from_numpy(self, values)
            return

        shp = self.shape()
        n = _elem_count(shp)
        vals = list(values)
        if len(vals) != n:
            raise ValueError(f"expected {n} values for shape {shp}, got {len(vals)}")

        buf = ffi.new("float[]", [float(v) for v in vals])
        st = lib.aion_tensor_write(self._t, int(AionDType.AION_DTYPE_F32), buf, n)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_tensor_write")

    def write_i32(self, values: Iterable[int]) -> None:
        if self.dtype() != AionDType.AION_DTYPE_I32:
            raise NotImplementedError("dtype mismatch: expected i32")

        shp = self.shape()
        n = _elem_count(shp)
        vals = list(values)
        if len(vals) != n:
            raise ValueError(f"expected {n} values for shape {shp}, got {len(vals)}")

        buf = ffi.new("int32_t[]", [int(v) for v in vals])
        st = lib.aion_tensor_write(self._t, int(AionDType.AION_DTYPE_I32), buf, n)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_tensor_write")

    def read_scalar_f32(self) -> float:
        if self.dtype() != AionDType.AION_DTYPE_F32:
            raise NotImplementedError("dtype mismatch: expected f32")

        out = ffi.new("float*")
        st = lib.aion_tensor_read_scalar(self._t, int(AionDType.AION_DTYPE_F32), out)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_tensor_read_scalar")
        return float(out[0])

    def read_scalar_i32(self) -> int:
        if self.dtype() != AionDType.AION_DTYPE_I32:
            raise NotImplementedError("dtype mismatch: expected i32")

        out = ffi.new("int32_t*")
        st = lib.aion_tensor_read_scalar(self._t, int(AionDType.AION_DTYPE_I32), out)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_tensor_read_scalar")
        return int(out[0])

    def item_f32(self) -> float:
        """Convenience: return the only element as a Python float (f32 only)."""
        return self.read_scalar_f32()

    def fill(self, value: float) -> "Tensor":
        """In-place fill (f32 only). Returns self."""

        if self.dtype() != AionDType.AION_DTYPE_F32:
            raise NotImplementedError("only f32 tensor I/O is supported by the current C ABI")
        n = self.numel()
        self.write_f32([float(value)] * n)
        return self

    def zero(self) -> "Tensor":
        """In-place zero fill. Returns self."""

        dt = self.dtype()
        if dt == AionDType.AION_DTYPE_F32:
            return self.fill(0.0)

        n = self.numel()
        if dt == AionDType.AION_DTYPE_F16:
            # `aion_tensor_write` expects element-count, not byte-count.
            buf = ffi.new("uint16_t[]", n)  # zero-initialized; IEEE-754 half 0.0.
            st = lib.aion_tensor_write(self._t, int(AionDType.AION_DTYPE_F16), buf, n)
            raise_for_status(st, self._ctx_owner.ptr, what="aion_tensor_write")
            return self
        if dt == AionDType.AION_DTYPE_I32:
            buf = ffi.new("int32_t[]", n)
            st = lib.aion_tensor_write(self._t, int(AionDType.AION_DTYPE_I32), buf, n)
            raise_for_status(st, self._ctx_owner.ptr, what="aion_tensor_write")
            return self

        raise NotImplementedError(f"zero() not implemented for dtype={dt}")

    # Optional numpy interop (enabled when numpy is installed).
    def numpy(self) -> NDArrayF32:
        from .numpy import tensor_to_numpy

        return tensor_to_numpy(self)

    @classmethod
    def from_numpy(cls, ctx, array: ArrayLike, *, device: DeviceLike = None) -> "Tensor":
        from .numpy import tensor_from_numpy

        t = tensor_from_numpy(ctx, array)
        if device is not None:
            t.to(device)
        return t

    def write_from_numpy(self, array: ArrayLike) -> None:
        from .numpy import tensor_write_from_numpy

        tensor_write_from_numpy(self, array)
