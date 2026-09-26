# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional, Sequence, cast

from .device import DeviceLike, _device_to_str, _normalize_device
from .dtype import (
    dtype_name,
    float32,
    is_quantized as _is_quant,
    normalize_dtype,
    numpy_dtype,
    q8_0,
)
from .context import Context
from ._ffi.dlpack import dtype_kind, empty_view, export_capsule, host_view
from ._ffi.handles import TensorHandle
from ._ffi.runtime import (
    create_empty_tensor,
    destroy_tensor,
    move_tensor,
    quantize_tensor,
    read_view,
    tensor_device,
    tensor_dtype,
    tensor_from_dlpack,
    tensor_shape,
    tensor_to_dlpack,
    write_view,
    zero_tensor,
)
from .enums import AionDeviceKind, AionDType
from .types import ArrayLike, DTypeLike, NDArray

type TensorData = int | float | list[TensorData]

if TYPE_CHECKING:
    from typing_extensions import CapsuleType

    from .builder import Builder, TensorRef

_KDL_CPU = 1


def _as_shape(shape: Sequence[int]) -> tuple[int, ...]:
    shp = tuple(int(x) for x in shape)
    if any(d < 0 for d in shp):
        raise ValueError("shape dims must be >= 0")
    return shp


class Tensor:
    """Concrete tensor data: owns an `AionTensor*`; storage belongs to the Context.

    This is the *data* handle — what you write inputs into, read outputs out of,
    and hand to a layer as a weight. It is deliberately not a graph node: graphs
    are built from `TensorRef`s on a `Builder`, which is the single graph
    representation (and the same split the Zig API makes between `api.Tensor` and
    `TensorRef`).

    Data comes in from anything exporting ``__dlpack__`` (numpy, PyTorch, JAX, ...)
    or from Python numbers and nested sequences, and goes out, without copying, to
    any DLPack consumer: ``np.from_dlpack(t)`` / ``torch.from_dlpack(t)`` give a
    read-only view of the tensor's bytes as they are now. `numpy` and `tolist`
    copy.
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
        """Create a tensor from host data.

        Examples:
            Tensor(np_f32_array)                     # borrowed: no copy
            Tensor(torch_bf16_weight)                # converted once -> float32
            Tensor([1, 2, 3])                        # int32
            Tensor([[1, 2], [3, 4]], dtype=aion.float32)
            Tensor([1, 2, 3], device="gpu")          # built on CPU, then migrated

        The dtype follows the data unless `dtype` is given: float16 and int8 stay,
        other floats become float32 and other integers int32.

        A DLPack exporter's memory is BORROWED when it already is the tensor's (its
        dtype, row-major): nothing is copied, and the exporter's later writes show
        through. Aion never writes it; a write to the tensor (or a model run into
        it) goes to a private copy first. Anything else is converted in one copy.

        `device` migrates the tensor after it is built on the CPU (move semantics:
        the host bytes are let go). Host conversion and mutation then fail until
        you migrate back with ``.to("cpu")``.
        """
        from .context import get_default_context

        if ctx is None:
            ctx = get_default_context()
        want = normalize_dtype(dtype) if dtype is not None else None
        if want is not None and _is_quant(want):
            raise NotImplementedError("use Tensor.quantize for quantized tensors")
        view = host_view(data, want)
        # The core borrows or converts; one of our own packed buffers (float64 or
        # int64, never a tensor dtype) is always converted, so it is never kept.
        with view.transfer() as managed:
            handle = tensor_from_dlpack(ctx.ptr, managed, want)
        self._init_from_handle(ctx, handle)
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

    # --- placement ---------------------------------------------------------

    def to(self, device: DeviceLike) -> "Tensor":
        """Migrate this tensor to `device` (move semantics; returns self).

        The source-device copy is freed. Migrating off the CPU makes host reads
        and writes fail until you migrate back with ``.to("cpu")``. Idempotent
        when already on the target device. Accepts ``"cpu"``, ``"gpu"``,
        ``"gpu:N"``, ``(kind, index)``, or an `AionDeviceKind`.
        """
        kind, index = _normalize_device(device)
        move_tensor(self._ctx_owner.ptr, self._require_handle(), int(kind), int(index))
        return self

    def device(self) -> str:
        """The device this tensor is resident on: ``"cpu"`` or ``"gpu:N"``."""
        kind, index = tensor_device(self._ctx_owner.ptr, self._require_handle())
        return _device_to_str(kind, index)

    # --- lifetime ----------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        handle = self._t
        # A context closes its children first; one closed after it (a finalizer at
        # interpreter exit) points into freed storage and must not call in.
        if handle is not None and not self._ctx_owner._closed:
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
                self.close()
        except Exception:
            pass

    # --- construction ------------------------------------------------------

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
        dt = normalize_dtype(dtype)
        t = cls._from_handle(ctx, create_empty_tensor(ctx.ptr, dt, shp), dtype=dt, shape=shp)
        if device is not None:
            t.to(device)
        return t

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
        """Quantize float `values` of `shape` into a packed-quant tensor.

        The core does the packing, blocking along `quant_axis`, which defaults to
        the matmul-B reduction axis (rank-2, i.e. the `K` of a `[…, K, N]` weight);
        pass the last axis for an embedding table blocked along its feature dim.
        `values` is read in place when it exports ``__dlpack__`` (any float dtype,
        bfloat16 included, strided), else packed from Python numbers.
        """
        dt = normalize_dtype(dtype)
        if not _is_quant(dt):
            raise ValueError(f"quantize expects a quantized dtype, got {dtype_name(dt)}")
        shp = _as_shape(shape)
        if quant_axis is None:
            quant_axis = len(shp) - 2 if len(shp) >= 2 else 0
        if not (0 <= quant_axis < len(shp)):
            raise ValueError(f"quant_axis {quant_axis} out of range for rank {len(shp)}")
        view = host_view(values, AionDType.AION_DTYPE_F32).fit(shp, reshape=True)
        handle = quantize_tensor(ctx.ptr, dt, quant_axis, view)
        return cls._from_handle(ctx, handle, dtype=dt, shape=shp)

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

        If `ctx` is omitted, the process-wide default context is used. With
        `device`, the tensor is zeroed on the CPU and then migrated.
        """
        from .context import get_default_context

        if ctx is None:
            ctx = get_default_context()
        t = cls.empty(ctx, shape, dtype=dtype)
        t.zero()
        if device is not None:
            t.to(device)
        return t

    # --- metadata ----------------------------------------------------------

    def numel(self) -> int:
        return math.prod(self.shape)

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
        if self._shape_cache is None:
            self._shape_cache = tensor_shape(self._ctx_owner.ptr, self._require_handle())
        return self._shape_cache

    # --- host reads and writes ----------------------------------------------

    def _flat_values(self) -> list[int | float]:
        if _is_quant(self.dtype):
            raise NotImplementedError(
                f"{dtype_name(self.dtype)}: quantized tensors cannot be converted to Python values"
            )
        buf, view = empty_view(self.shape, dtype_kind(self.dtype))
        read_view(self._ctx_owner.ptr, self._require_handle(), view)
        return list(buf[0 : self.numel()])

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
        """Copy host data into this tensor and return ``self``.

        `values` has the tensor's shape, or is a single element broadcast to it.
        It converts to the tensor's dtype as `Tensor(...)` data does.
        """
        dt = self.dtype
        if _is_quant(dt):
            raise NotImplementedError(f"{dtype_name(dt)}: quantized tensors cannot be written")
        try:
            view = host_view(values, dt).fit(self.shape)
        except ValueError as e:
            raise ValueError(f"shape mismatch: tensor {self.shape}: {e}") from None
        write_view(self._ctx_owner.ptr, self._require_handle(), view)
        return self

    def fill(self, value: int | float) -> "Tensor":
        """In-place scalar fill. Returns self."""
        return self.copy_from(value)

    def zero(self) -> "Tensor":
        """In-place zero fill, wherever the tensor lives. Returns self."""
        zero_tensor(self._ctx_owner.ptr, self._require_handle())
        return self

    def numpy(self) -> NDArray:
        """A copy of the tensor as a new numpy array (scalar dtypes only).

        For a view of the tensor's bytes without the copy, use
        ``np.from_dlpack(tensor)`` (read-only).
        """
        try:
            import numpy as np
        except ImportError as e:  # pragma: no cover
            raise ImportError("NumPy is required; install aion-engine[numpy]") from e
        if _is_quant(self.dtype):
            raise NotImplementedError(f"{dtype_name(self.dtype)}: quantized tensors have no numpy representation")
        out = np.empty(self.shape, dtype=numpy_dtype(self.dtype))
        read_view(self._ctx_owner.ptr, self._require_handle(), host_view(out))
        return out

    # --- DLPack export -------------------------------------------------------

    def __dlpack__(
        self,
        *,
        stream: Any = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> CapsuleType:
        """Export the tensor's bytes to a DLPack consumer, without copying.

        The export is read-only and stays valid, showing the values as they are
        now, whatever later happens to the tensor: a later write (or a model run
        that produces it) goes to a fresh copy, and the tensor may be closed first.
        Host tensors of scalar dtypes only; ``copy=True`` exports a private copy.
        """
        if stream is not None:
            raise BufferError("an Aion host tensor takes no stream")
        if max_version is None or max_version[0] < 1:
            raise BufferError("Aion exports DLPack 1.x (versioned) capsules only")
        if dl_device is not None and tuple(dl_device) != (_KDL_CPU, 0):
            raise BufferError(f"cannot export to device {dl_device}; migrate the tensor instead")
        src = self
        if copy:
            src = Tensor.empty(self._ctx_owner, self.shape, dtype=self.dtype)
            try:
                src.copy_from(self)
            except BaseException:
                src.close()
                raise
        try:
            managed = tensor_to_dlpack(self._ctx_owner.ptr, src._require_handle())
        finally:
            if src is not self:
                src.close()  # the export shares its bytes and outlives it
        return export_capsule(managed)

    def __dlpack_device__(self) -> tuple[int, int]:
        kind, index = tensor_device(self._ctx_owner.ptr, self._require_handle())
        # A GPU tensor reports where it is (WebGPU) and refuses to export.
        return (_KDL_CPU, 0) if kind == int(AionDeviceKind.AION_DEVICE_CPU) else (15, index)


def from_dlpack(
    data: ArrayLike,
    *,
    ctx: "Context | None" = None,
    dtype: DTypeLike | None = None,
    device: DeviceLike = None,
) -> Tensor:
    """A tensor of a DLPack exporter's data (numpy, PyTorch, ...): its memory
    borrowed, not copied, when it already is the tensor's (see `Tensor`)."""
    if not hasattr(data, "__dlpack__"):
        raise TypeError(f"{type(data).__name__} does not export __dlpack__")
    return Tensor(data, ctx=ctx, dtype=dtype, device=device)
