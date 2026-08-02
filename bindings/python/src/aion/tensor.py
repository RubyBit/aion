# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import struct
from types import TracebackType
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, cast

from .device import DeviceLike, _device_to_str, _normalize_device
from .dtype import (
    dtype_name,
    float32,
    is_quantized as _is_quant,
    normalize_dtype,
    q8_0,
)
from .context import Context
from ._ffi.handles import TensorHandle
from ._ffi.runtime import (
    create_empty_tensor,
    create_empty_tiled_tensor,
    create_tensor,
    destroy_tensor,
    move_tensor,
    quantize_tensor,
    read_tensor,
    tensor_device,
    tensor_dtype,
    tensor_shape,
    write_tensor,
    zero_tensor,
)
from .enums import AionDType
from .types import ArrayLike, DTypeLike, NDArray

type TensorData = int | float | list[TensorData]

if TYPE_CHECKING:
    from .builder import Builder, TensorRef
    from .model import LoadedModel

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
    if isinstance(data, (int, float)):
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


def _try_numpy():
    try:
        import numpy as np

        return np
    except ImportError:
        return None


def _native_values(
    dtype: AionDType, values: Iterable[int | float]
) -> list[int | float]:
    """Convert Python numbers to the host buffer representation for ``dtype``."""
    if dtype == AionDType.AION_DTYPE_F32:
        return [float(value) for value in values]
    if dtype == AionDType.AION_DTYPE_F16:
        return [
            struct.unpack("=H", struct.pack("=e", float(value)))[0]
            for value in values
        ]
    if dtype in (AionDType.AION_DTYPE_I8, AionDType.AION_DTYPE_I32):
        return [int(value) for value in values]
    raise NotImplementedError(
        f"{dtype_name(dtype)} has no per-element host write path"
    )


def _flatten_for_dtype(
    data: ArrayLike, dtype: AionDType
) -> tuple[tuple[int, ...], list[int | float]]:
    if dtype in (AionDType.AION_DTYPE_F32, AionDType.AION_DTYPE_F16):
        return _flatten_nested(data)
    if dtype in (AionDType.AION_DTYPE_I8, AionDType.AION_DTYPE_I32):
        shape, values = _flatten_nested_i32(data)
        return shape, cast(list[int | float], values)
    raise NotImplementedError(
        f"{dtype_name(dtype)} has no per-element host write path"
    )


def _infer_construct_dtype(data) -> AionDType:
    """Default construction dtype: I32 for integer data, otherwise F32."""
    np = _try_numpy()
    if np is not None:
        try:
            array_dtype = np.asarray(data).dtype
        except (TypeError, ValueError):
            pass
        else:
            if np.issubdtype(array_dtype, np.integer):
                return AionDType.AION_DTYPE_I32
    if isinstance(data, int) and not isinstance(data, bool):
        return AionDType.AION_DTYPE_I32
    if isinstance(data, (list, tuple)) and data:
        if all(_infer_construct_dtype(item) == AionDType.AION_DTYPE_I32 for item in data):
            return AionDType.AION_DTYPE_I32
    return AionDType.AION_DTYPE_F32


def _is_ndarray(x) -> bool:
    np = _try_numpy()
    return np is not None and isinstance(x, np.ndarray)


class Tensor:
    """Concrete tensor data: owns an `AionTensor*`; storage belongs to the Context.

    This is the *data* handle — what you write inputs into, read outputs out of,
    and hand to a layer as a weight. It is deliberately not a graph node: graphs
    are built from `TensorRef`s on a `Builder`, which is the single graph
    representation (and the same split the Zig API makes between `api.Tensor` and
    `TensorRef`).
    """

    # Instance attributes (assigned in `_init_from_handle`; declared here so type
    # checkers see them).
    _ctx_owner: "Context"
    _closed: bool
    _dtype_cache: Optional[AionDType]
    _shape_cache: Optional[tuple[int, ...]]
    _t: TensorHandle | None          # typed opaque native handle
    _name: Optional[str]             # parameter name carried into a graph

    def __init__(
        self,
        data: ArrayLike,
        *,
        ctx: "Context | None" = None,
        dtype: DTypeLike | None = None,
        device: DeviceLike = None,
    ) -> None:
        """Create a tensor from Python data.

        Examples:
            Tensor([1, 2, 3])
            Tensor([[1, 2], [3, 4]], dtype=aion.float32)
            Tensor([1, 2, 3], device="gpu")   # built on CPU, then migrated

        `device` migrates the tensor after it is built and written on the CPU
        (move semantics: the host copy is freed). Host conversion and mutation
        then fail until you migrate back with ``.to("cpu")``.
        """

        from .context import get_default_context

        if ctx is None:
            ctx = get_default_context()
        if dtype is None:
            dtype = _infer_construct_dtype(data)
        else:
            dtype = normalize_dtype(dtype)
        if dtype in (AionDType.AION_DTYPE_Q4_0, AionDType.AION_DTYPE_Q8_0):
            raise NotImplementedError("quantized tensor construction is not supported by the current C ABI")

        np = _try_numpy()
        if np is not None and isinstance(data, np.ndarray):
            from .numpy import _tensor_from_numpy

            self._take(_tensor_from_numpy(ctx, data, dtype=dtype))
            if device is not None:
                self.to(device)
            return
        if np is not None and dtype not in (
            AionDType.AION_DTYPE_F32,
            AionDType.AION_DTYPE_I32,
        ):
            from .numpy import _tensor_from_numpy

            self._take(_tensor_from_numpy(ctx, data, dtype=dtype))
            if device is not None:
                self.to(device)
            return

        shape, vals = _flatten_for_dtype(data, dtype)

        shp = _as_shape(shape)
        n = _elem_count(shp)
        if len(vals) != n:
            raise ValueError(f"expected {n} values for shape {shp}, got {len(vals)}")

        # The native runtime currently represents a scalar as shape (1,).
        if len(shp) == 0:
            shp = [1]
        native_values = _native_values(dtype, vals)

        handle = create_tensor(ctx.ptr, dtype, shp, native_values)
        self._init_from_handle(ctx, handle, dtype=dtype, shape=tuple(shp))

        if device is not None:
            self.to(device)

    def _init_from_handle(
        self,
        ctx: "Context",
        ptr: TensorHandle,
        *,
        dtype: AionDType | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> None:
        self._ctx_owner = ctx
        self._ctx_owner._register_child(self)
        self._t = ptr
        self._closed = False
        self._dtype_cache = dtype
        self._shape_cache = shape
        # A concrete tensor keeps whatever name it was given: lowered into a graph
        # it becomes a param, and a param's name is how a loaded model finds it.
        if not hasattr(self, "_name"):
            self._name = None

    def _take(self, other: "Tensor") -> None:
        """Adopt `other`'s handle, leaving it closed. Used by the numpy paths,
        which build a finished tensor that `__init__` must become."""
        handle = other._require_handle()
        self._init_from_handle(
            other._ctx_owner,
            handle,
            dtype=other._dtype_cache,
            shape=other._shape_cache,
        )
        other._t = None
        other._closed = True


    @classmethod
    def _from_handle(
        cls,
        ctx: "Context",
        ptr: TensorHandle,
        *,
        dtype: AionDType | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> "Tensor":
        self = cls.__new__(cls)
        self._init_from_handle(ctx, ptr, dtype=dtype, shape=shape)
        return self


    @property
    def ptr(self) -> TensorHandle:
        return self._require_handle()

    def ref(self, builder: "Builder | None" = None) -> "TensorRef":
        """Bind this data into a graph and return the handle to compose on.

        `Tensor` is data and has no operators — the graph lives on a `Builder`. This
        is the one step across that line:

            x = aion.tensor(np_x).ref()
            print((x @ w.ref()).relu())

        With no `builder`, it uses the Context's scratch builder, so exploring costs
        no ceremony. Pass an explicit builder when authoring a model, so that model
        owns its graph.

        The name carried by `rename()` becomes the parameter name, which is how a
        loaded model looks the weight up.
        """
        b = builder if builder is not None else self._ctx_owner.scratch_builder()
        if self._name is not None:
            return b.param_named(self, self._name)
        return b.param(self)

    def _require_handle(self) -> TensorHandle:
        if self._t is None:
            raise RuntimeError("Tensor has no concrete native handle")
        return self._t






















    def rename(self, name: str) -> "Tensor":
        """Set the name this tensor carries into a graph.

        This is the *parameter* name a loaded model looks the weight up by, so
        naming weights is how a model gets a usable `state_dict`.
        """
        self._name = name
        return self

    @property
    def ndim(self) -> int:
        return len(self.shape)

    # --- materialization ---------------------------------------------------


    def to(self, device: DeviceLike) -> "Tensor":
        """Migrate this tensor to `device` (move semantics; returns self).

        The source-device copy is freed. Migrating off the CPU makes host
        `read_*`/`write_*` fail until you migrate back with ``.to("cpu")``.
        Idempotent when already on the target device. Accepts ``"cpu"``,
        ``"gpu"``, ``"gpu:N"``, ``(kind, index)``, or an `AionDeviceKind``.
            """

        kind, index = _normalize_device(device)
        move_tensor(
            self._ctx_owner.ptr, self._require_handle(), int(kind), int(index)
        )
        return self

    def device(self) -> str:
        """The device this tensor is resident on: ``"cpu"`` or ``"gpu:N"``."""

        kind, index = tensor_device(self._ctx_owner.ptr, self._require_handle())
        return _device_to_str(kind, index)

    def close(self) -> None:
        if self._closed:
            return
        handle = self._t
        if handle is not None:
            destroy_tensor(handle)
        try:
            self._ctx_owner._unregister_child(self)
        except Exception:
            pass
        self._t = None
        self._closed = True

    def __enter__(self) -> "Tensor":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover
        try:
            if not getattr(self, "_closed", True):
                handle = getattr(self, "_t", None)
                if handle is not None:
                    destroy_tensor(handle)
        except Exception:
            pass

    @classmethod
    def empty(
        cls,
        ctx: "Context",
        shape: Sequence[int],
        *,
        dtype: DTypeLike = float32,
        device: DeviceLike = None,
    ) -> "Tensor":
        shp = _as_shape(shape)
        dtype = normalize_dtype(dtype)

        handle = create_empty_tensor(ctx.ptr, dtype, shp)
        t = cls._from_handle(ctx, handle, dtype=dtype, shape=tuple(shp))
        if device is not None:
            t.to(device)
        return t

    @classmethod
    def empty_tiled(
        cls,
        ctx: "Context",
        shape: Sequence[int],
        tile_shape: Sequence[int],
        *,
        dtype: DTypeLike = float32,
    ) -> "Tensor":
        """Create an empty tensor with an explicit per-axis tile shape.

        Use this when you want to pin a specific tile layout — e.g. to skip the
        one-time retile copy the compiler would otherwise insert when an op needs
        a different tiling. For most cases `Tensor.empty` is sufficient: the graph
        compiler retiles inputs as needed (e.g. KV caches consumed by
        `SequenceAppend` are coerced to head-dim-contiguous on first run).
        """
        shp = _as_shape(shape)
        tshp = _as_shape(tile_shape)
        if len(tshp) != len(shp):
            raise ValueError(f"tile_shape rank {len(tshp)} != shape rank {len(shp)}")
        dtype = normalize_dtype(dtype)

        handle = create_empty_tiled_tensor(ctx.ptr, dtype, shp, tshp)
        return cls._from_handle(ctx, handle, dtype=dtype, shape=tuple(shp))

    @classmethod
    def quantize(
        cls,
        ctx: "Context",
        shape: Sequence[int],
        values: ArrayLike,
        *,
        dtype: DTypeLike = q8_0,
        quant_axis: int | None = None,
    ) -> "Tensor":
        """Quantize row-major f32 `values` into a packed-quant tensor.

        The core does the packing (q8_0 today), blocking along `quant_axis`.
        `quant_axis` defaults to the matmul-B reduction axis (rank-2, i.e. the
        `K` of a `[…, K, N]` weight); pass the last axis for an embedding table
        blocked along its feature dim. `values` may be a numpy array or a
        (possibly nested) Python sequence.
        """
        dtype = normalize_dtype(dtype)
        if dtype not in (AionDType.AION_DTYPE_Q8_0, AionDType.AION_DTYPE_Q4_0):
            raise ValueError(
                f"quantize expects a quantized dtype, got {dtype_name(dtype)}"
            )

        shp = _as_shape(shape)
        if quant_axis is None:
            quant_axis = len(shp) - 2 if len(shp) >= 2 else 0
        if not (0 <= quant_axis < len(shp)):
            raise ValueError(f"quant_axis {quant_axis} out of range for rank {len(shp)}")
        n = _elem_count(shp)

        try:
            import numpy as np
        except ImportError:
            np = None

        if np is not None and isinstance(values, np.ndarray):
            arr = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
            if arr.size != n:
                raise ValueError(f"expected {n} values for shape {shp}, got {arr.size}")
            native_values: object = arr
            from_buffer = True
        else:
            _, flat = _flatten_nested(values)
            if len(flat) != n:
                raise ValueError(f"expected {n} values for shape {shp}, got {len(flat)}")
            native_values = [float(v) for v in flat]
            from_buffer = False

        handle = quantize_tensor(
            ctx.ptr,
            dtype,
            shp,
            quant_axis,
            native_values,
            n,
            from_buffer=from_buffer,
        )
        return cls._from_handle(ctx, handle, dtype=dtype, shape=tuple(shp))

    @classmethod
    def zeros(
        cls,
        shape: Sequence[int],
        *,
        ctx: "Context | None" = None,
        dtype: DTypeLike = float32,
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
    def _from_flat(
        cls,
        ctx: "Context",
        shape: Sequence[int],
        values: Iterable[int | float],
        dtype: AionDType,
    ) -> "Tensor":
        shp = _as_shape(shape)
        n = _elem_count(shp)
        vals = list(values)
        if len(vals) != n:
            raise ValueError(f"expected {n} values for shape {shp}, got {len(vals)}")
        if len(shp) == 0:
            shp = [1]
        handle = create_tensor(ctx.ptr, dtype, shp, _native_values(dtype, vals))
        return cls._from_handle(ctx, handle, dtype=dtype, shape=tuple(shp))

    def numel(self) -> int:
        return _elem_count(self.shape)

    def __repr__(self) -> str:
        if getattr(self, "_closed", True):
            return "Tensor(<closed>)"
        try:
            return f"Tensor(shape={self.shape}, dtype={dtype_name(self.dtype)})"
        except Exception:
            return "Tensor(<uninitialized>)"

    @property
    def dtype(self) -> AionDType:
        if self._dtype_cache is None:
            self._dtype_cache = tensor_dtype(self._require_handle())
        return self._dtype_cache


    @property
    def shape(self) -> tuple[int, ...]:
        if self._shape_cache is not None:
            return self._shape_cache

        self._shape_cache = tensor_shape(
            self._ctx_owner.ptr, self._require_handle()
        )
        return self._shape_cache

    # --- host conversion and mutation ------------------------------------
    def _flat_values(self) -> list[int | float]:
        if _is_quant(self.dtype):
            raise NotImplementedError(
                f"{dtype_name(self.dtype)}: quantized tensors cannot be converted to Python values"
            )
        n = _elem_count(self.shape)
        return read_tensor(self._ctx_owner.ptr, self._require_handle(), n)

    def tolist(self) -> TensorData:
        """Return the tensor as nested Python lists."""
        values = self._flat_values()

        def build(shape: tuple[int, ...], offset: int = 0) -> tuple[object, int]:
            if not shape:
                return values[offset], offset + 1
            items: list[object] = []
            for _ in range(shape[0]):
                item, offset = build(shape[1:], offset)
                items.append(item)
            return items, offset

        result, _ = build(self.shape)
        return cast(TensorData, result)

    def item(self) -> int | float:
        """Return the value of a one-element tensor as a Python scalar."""
        if self.numel() != 1:
            raise ValueError(f"item() requires one element, got {self.numel()}")
        return self._flat_values()[0]

    def copy_from(self, values: ArrayLike) -> "Tensor":
        """Copy data into this tensor, broadcasting a scalar, and return ``self``.

        Non-scalar inputs must match the tensor's exact shape.
        """
        dt = self.dtype
        if _is_quant(dt):
            raise NotImplementedError(
                f"{dtype_name(dt)}: quantized tensors cannot be mutated"
            )

        np = _try_numpy()
        if np is not None:
            from .numpy import _copy_numpy_into_tensor

            _copy_numpy_into_tensor(self, values)
            return self

        shape, vals = _flatten_for_dtype(values, dt)
        if shape == ():
            vals *= self.numel()
        elif shape != self.shape:
            raise ValueError(f"shape mismatch: tensor {self.shape} vs data {shape}")
        write_tensor(
            self._ctx_owner.ptr,
            self._require_handle(),
            _native_values(dt, vals),
        )
        return self

    def fill(self, value: int | float) -> "Tensor":
        """In-place scalar fill. Returns self."""
        return self.copy_from(value)

    def zero(self) -> "Tensor":
        """In-place zero fill (scalar dtypes). Returns self."""
        dt = self.dtype
        if _is_quant(dt):
            raise NotImplementedError(
                f"zero() not implemented for dtype={dtype_name(dt)}"
            )
        n = self.numel()
        # The native façade zero-initializes the temporary C buffer; for f16,
        # uint16 zero is IEEE-754 +0.0.
        zero_tensor(self._ctx_owner.ptr, self._require_handle(), n)
        return self

    # numpy interop (requires numpy; scalar dtypes only).
    def numpy(self) -> NDArray:
        from .numpy import _tensor_to_numpy

        return _tensor_to_numpy(self)
