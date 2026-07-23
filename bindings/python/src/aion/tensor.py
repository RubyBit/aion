# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from types import TracebackType
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, cast

from .device import DeviceLike, _device_to_str, _normalize_device
from .dtype import is_quantized as _is_quant, normalize_dtype
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
from .types import NDArray

type TensorData = int | float | list[TensorData]

if TYPE_CHECKING:
    from .builder import Builder, Value
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


def _try_numpy():
    try:
        import numpy as np

        return np
    except ImportError:
        return None


def _is_py_scalar(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


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


# --- lazy DAG lowering ----------------------------------------------------
# A lazy `Tensor` is an immutable node `(op, inputs, attrs)`; concrete tensors
# and free inputs are its leaves. Nothing touches the core until we *lower* the
# reachable DAG into a throwaway `Builder` (walked once, memoized), which the
# core validates + shape-infers. Shape/dtype, realize, and compile/export all go
# through `_lower`. Because each lowering builds a fresh builder from the
# immutable DAG, there is no shared/destructible graph state — tensors combine
# freely, realize any time, re-realize at will.

def _walk(outputs):
    """Yield every distinct Tensor reachable from `outputs` (self + inputs)."""
    seen: set[int] = set()
    stack = list(outputs)
    while stack:
        t = stack.pop()
        if not isinstance(t, Tensor) or id(t) in seen:
            continue
        seen.add(id(t))
        yield t
        if t._lazy:
            stack.extend(i for i in t._inputs if isinstance(i, Tensor))


def _resolve_ctx(outputs):
    ctx = None
    for t in _walk(outputs):
        c = t._ctx_owner
        if ctx is None:
            ctx = c
        elif c is not ctx:
            raise ValueError("tensors belong to different contexts")
    if ctx is None:
        raise ValueError("no context found for lazy tensor(s)")
    return ctx


def _has_free_input(outputs) -> bool:
    return any(t._lazy and t._op == "input" for t in _walk(outputs))


def _scalar_const_value(value, sibling, b):
    """A scalar `Value` broadcast to `sibling`'s shape (elemwise won't broadcast
    a `(1,)`), dtype following `sibling`."""
    dt = sibling.dtype
    shape = sibling.shape
    if dt == AionDType.AION_DTYPE_I32:
        if isinstance(value, float) and not float(value).is_integer():
            raise TypeError("cannot combine a non-integer scalar with an i32 tensor; cast explicitly")
        np_dtype, val = "int32", int(value)
    elif dt == AionDType.AION_DTYPE_F32:
        np_dtype, val = "float32", float(value)
    else:
        raise TypeError(f"scalar arithmetic is not supported for dtype {dt.name}; cast explicitly")

    np = _try_numpy()
    if np is not None:
        const = Tensor(np.full(shape, val, dtype=np_dtype), ctx=b.context)
    else:
        const = Tensor._from_flat(
            b.context,
            shape,
            [float(val)] * _elem_count(shape),
            AionDType.AION_DTYPE_F32,
        )
    return b.param(const)


def _lower(outputs) -> "tuple[Builder, list[Value]]":
    """Emit the DAG reachable from `outputs` into a fresh `Builder`.

    Returns `(builder, [Value per output])`. Each node is emitted once
    (memoized); a lazy node caches its inferred shape/dtype so later `.shape`
    queries are free.
    """
    from .builder import Builder, Value

    ctx = _resolve_ctx(outputs)
    b = Builder(ctx=ctx)
    memo: dict[int, "Value"] = {}

    def operand(x):
        # A raw scalar has no Value on its own; the caller (elemwise) handles it.
        if isinstance(x, Tensor):
            return emit(x)
        if _is_ndarray(x):
            return b.param(Tensor(x, ctx=ctx))
        raise TypeError(f"unsupported operand for a tensor op: {type(x).__name__}")

    def emit(t):
        k = id(t)
        cached = memo.get(k)
        if cached is not None:
            return cached
        if not t._lazy:
            v = b.param(t)  # concrete leaf -> baked param
        else:
            v = _emit_node(t, b, emit, operand)
            if t._shape_cache is None:
                t._shape_cache = tuple(v.shape)
                t._dtype_cache = v.dtype
            if t._name:
                b.name(v, t._name)
        memo[k] = v
        return v

    out_values = [emit(t) for t in outputs]
    return b, out_values


def _emit_node(t, b, emit, operand):
    op, ins, at = t._op, t._inputs, t._attrs
    if op == "input":
        return b.input(at["shape"], dtype=at["dtype"], dynamic=at.get("dynamic"))
    if op == "matmul":
        return b.matmul(operand(ins[0]), operand(ins[1]), at.get("alpha", 1.0), at.get("beta", 0.0))
    if op == "unary":
        return b.unary(at["op"], operand(ins[0]))
    if op == "softmax":
        return b.softmax(operand(ins[0]), at["axis"])
    if op == "reshape":
        return b.reshape(operand(ins[0]), at["shape"])
    if op == "transpose2d":
        return b.transpose2d(operand(ins[0]))
    if op == "cast":
        return b.cast(operand(ins[0]), at["dtype"])
    if op in ("add", "sub", "mul", "div"):
        xv, yv = _emit_pair(ins[0], ins[1], b, emit, operand)
        return getattr(b, op)(xv, yv)
    raise ValueError(f"unknown lazy op {op!r}")


def _emit_pair(x, y, b, emit, operand):
    xv = None if _is_py_scalar(x) else operand(x)
    yv = None if _is_py_scalar(y) else operand(y)
    if xv is None and yv is None:
        raise TypeError("a binary op needs at least one tensor operand")
    if xv is None:
        xv = _scalar_const_value(x, yv, b)
    if yv is None:
        yv = _scalar_const_value(y, xv, b)
    return xv, yv


def _realize_many(tensors, *, device: DeviceLike = None):
    """Lower + compile + run one throwaway graph, materializing each lazy tensor.

    Lazy tensors are flipped to CONCRETE in place, each keeping the `LoadedModel`
    (and its baked params) alive. Shared subgraphs are emitted once, so passing
    several tensors realizes them in a single compile + run.
    """
    lazies = [t for t in tensors if t._lazy]
    if not lazies:
        return list(tensors)
    if _has_free_input(lazies):
        raise RuntimeError(
            "cannot realize a graph with unbound inputs; build a runnable model "
            "with aion.compile(outputs, inputs=[...]) and bind them at run time"
        )

    b, out_values = _lower(lazies)
    names = []
    for i, (t, v) in enumerate(zip(lazies, out_values)):
        nm = t._name or f"__out{i}"
        b.mark_output(v, nm)
        names.append(nm)
    model = b.compile(device=device)
    model._attach_authoring_builder(b)  # keep baked params alive
    model.run()
    for t, nm in zip(lazies, names):
        h = model.output_tensor(nm)
        t._adopt(h)               # flip LAZY -> CONCRETE in place
        t._realize_model = model  # keep output storage alive
        t._op = None
        t._inputs = []            # release the DAG for GC
    return list(tensors)


class Tensor:
    """The high-level value: either CONCRETE data or a LAZY DAG node.

    - CONCRETE: owns an `AionTensor*` handle (`_t`); underlying storage is owned
      by the Context. Produced by `aion.tensor(data)` and the factories.
    - LAZY: an immutable graph node `(_op, _inputs, _attrs)`; owns no C storage.
      Produced by operators and `aion.tensor(shape=...)`. Materialized to
      CONCRETE (in place) by `.realize()` / `.numpy()` / `aion.realize`.
    """

    # Instance attributes (assigned in `_init_from_handle` / `_from_node`;
    # declared here so type checkers see them). `_lazy` selects the mode.
    _lazy: bool
    _ctx_owner: "Context"
    _closed: bool
    _dtype_cache: Optional[AionDType]
    _shape_cache: Optional[tuple[int, ...]]
    _t: TensorHandle | None          # CONCRETE: typed opaque native handle
    _op: Optional[str]               # LAZY: op name ("matmul", "add", "input", ...)
    _inputs: list[Any]               # LAZY: operands (Tensor | scalar | ndarray)
    _attrs: dict[str, Any]           # LAZY: op-specific attributes
    _name: Optional[str]             # LAZY: `.rename()` — default input/output name
    _realize_model: Optional["LoadedModel"]  # kept alive after realize (output storage)

    def __init__(
        self,
        data: object,
        *,
        ctx: "Context | None" = None,
        dtype: object | None = None,
        device: DeviceLike = None,
    ) -> None:
        """Create a tensor from Python data.

        Examples:
            Tensor([1, 2, 3])
            Tensor([[1, 2], [3, 4]], dtype=AionDType.AION_DTYPE_F32)
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

            self._adopt(_tensor_from_numpy(ctx, data, dtype=dtype))
            if device is not None:
                self.to(device)
            return
        if np is not None and dtype not in (
            AionDType.AION_DTYPE_F32,
            AionDType.AION_DTYPE_I32,
        ):
            from .numpy import _tensor_from_numpy

            self._adopt(_tensor_from_numpy(ctx, data, dtype=dtype))
            if device is not None:
                self.to(device)
            return

        if dtype == AionDType.AION_DTYPE_F32:
            shape, vals = _flatten_nested(data)
        elif dtype == AionDType.AION_DTYPE_I32:
            shape, vals = _flatten_nested_i32(data)
        else:
            raise NotImplementedError(f"tensor construction not implemented for dtype={dtype}")

        shp = _as_shape(shape)
        n = _elem_count(shp)
        if len(vals) != n:
            raise ValueError(f"expected {n} values for shape {shp}, got {len(vals)}")

        # The native runtime currently represents a scalar as shape (1,).
        if len(shp) == 0:
            shp = [1]
        if dtype == AionDType.AION_DTYPE_F32:
            native_values: list[int | float] = [float(v) for v in vals]
        elif dtype == AionDType.AION_DTYPE_I32:
            native_values = [int(v) for v in vals]
        else:
            raise NotImplementedError(f"tensor construction not implemented for dtype={dtype}")

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
        # CONCRETE mode: owns C storage, not a lazy DAG node.
        self._lazy = False

    def _adopt(self, other: "Tensor") -> None:
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

    @classmethod
    def _from_node(
        cls,
        op: str,
        inputs: list[Any],
        attrs: dict[str, Any],
        ctx: "Context",
        name: str | None = None,
    ) -> "Tensor":
        """Create a LAZY tensor: an immutable DAG node `(op, inputs, attrs)`.

        `inputs` may hold Tensors, Python scalars, or numpy arrays; leaves are
        resolved at lowering time. Owns no C storage — teardown is inert
        (`close`/`__del__` are no-ops). `realize()` flips it to CONCRETE in place.
        """
        self = cls.__new__(cls)
        self._lazy = True
        self._op = op
        self._inputs = list(inputs)
        self._attrs = attrs
        self._ctx_owner = ctx
        self._name = name
        self._t = None
        self._closed = True  # inert: no handle to free, not a context child
        self._dtype_cache = None
        self._shape_cache = None
        self._realize_model = None
        return self

    @property
    def ptr(self) -> TensorHandle:
        return self._require_handle()

    def _require_handle(self) -> TensorHandle:
        if self._t is None:
            raise RuntimeError("Tensor has no concrete native handle")
        return self._t

    # --- lazy op surface (build an immutable DAG; nothing runs yet) --------
    def _mk(self, kind: str, inputs: list, **attrs) -> "Tensor":
        return Tensor._from_node(kind, inputs, attrs, ctx=self._ctx_owner)

    def __matmul__(self, other: object) -> "Tensor":
        return self._mk("matmul", [self, other])

    def __rmatmul__(self, other: object) -> "Tensor":
        return self._mk("matmul", [other, self])

    def __add__(self, other: object) -> "Tensor":
        return self._mk("add", [self, other])

    def __radd__(self, other: object) -> "Tensor":
        return self._mk("add", [other, self])

    def __sub__(self, other: object) -> "Tensor":
        return self._mk("sub", [self, other])

    def __rsub__(self, other: object) -> "Tensor":
        return self._mk("sub", [other, self])

    def __mul__(self, other: object) -> "Tensor":
        return self._mk("mul", [self, other])

    def __rmul__(self, other: object) -> "Tensor":
        return self._mk("mul", [other, self])

    def __truediv__(self, other: object) -> "Tensor":
        return self._mk("div", [self, other])

    def __rtruediv__(self, other: object) -> "Tensor":
        return self._mk("div", [other, self])

    def __neg__(self) -> "Tensor":
        return self._mk("mul", [self, -1])

    def relu(self) -> "Tensor":
        return self._mk("unary", [self], op="relu")

    def gelu(self) -> "Tensor":
        return self._mk("unary", [self], op="gelu")

    def silu(self) -> "Tensor":
        return self._mk("unary", [self], op="silu")

    def sigmoid(self) -> "Tensor":
        return self._mk("unary", [self], op="sigmoid")

    def tanh(self) -> "Tensor":
        return self._mk("unary", [self], op="tanh")

    def softmax(self, axis: int = -1) -> "Tensor":
        return self._mk("softmax", [self], axis=axis)

    def reshape(self, shape: Sequence[int]) -> "Tensor":
        return self._mk("reshape", [self], shape=tuple(int(d) for d in shape))

    def transpose2d(self) -> "Tensor":
        return self._mk("transpose2d", [self])

    def cast(self, dtype: object) -> "Tensor":
        return self._mk("cast", [self], dtype=dtype)

    def rename(self, name: str) -> "Tensor":
        if not self._lazy:
            raise TypeError("rename() is only valid on a lazy tensor (input/op result)")
        self._name = name  # the default input/output name at lower/compile/export
        return self

    @property
    def ndim(self) -> int:
        return len(self.shape)

    # --- materialization ---------------------------------------------------
    def realize(self, *, device: DeviceLike = None) -> "Tensor":
        """Compile + run this lazy graph, becoming a CONCRETE tensor (returns self).

        A no-op on an already-CONCRETE tensor. To materialize several results
        from one graph in a single compile + run, use `aion.realize([...])`.
        """
        if self._lazy:
            _realize_many([self], device=device)
        return self

    def _ensure_realized(self) -> None:
        if self._lazy:
            _realize_many([self], device=None)

    def to(self, device: DeviceLike) -> "Tensor":
        """Migrate this tensor to `device` (move semantics; returns self).

        The source-device copy is freed. Migrating off the CPU makes host
        `read_*`/`write_*` fail until you migrate back with ``.to("cpu")``.
        Idempotent when already on the target device. Accepts ``"cpu"``,
        ``"gpu"``, ``"gpu:N"``, ``(kind, index)``, or an `AionDeviceKind``.
        A GRAPH (lazy) tensor is realized first (on the default device).
        """

        self._ensure_realized()
        kind, index = _normalize_device(device)
        move_tensor(
            self._ctx_owner.ptr, self._require_handle(), int(kind), int(index)
        )
        return self

    def device(self) -> str:
        """The device this tensor is resident on: ``"cpu"`` or ``"gpu:N"``."""

        self._ensure_realized()
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
        dtype: object | None = None,
        device: DeviceLike = None,
    ) -> "Tensor":
        shp = _as_shape(shape)
        dtype = AionDType.AION_DTYPE_F32 if dtype is None else normalize_dtype(dtype)

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
        dtype: object | None = None,
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
        dtype = AionDType.AION_DTYPE_F32 if dtype is None else normalize_dtype(dtype)

        handle = create_empty_tiled_tensor(ctx.ptr, dtype, shp, tshp)
        return cls._from_handle(ctx, handle, dtype=dtype, shape=tuple(shp))

    @classmethod
    def quantize(
        cls,
        ctx: "Context",
        shape: Sequence[int],
        values: object,
        *,
        dtype: object | None = None,
        quant_axis: int | None = None,
    ) -> "Tensor":
        """Quantize row-major f32 `values` into a packed-quant tensor.

        The core does the packing (q8_0 today), blocking along `quant_axis`.
        `quant_axis` defaults to the matmul-B reduction axis (rank-2, i.e. the
        `K` of a `[…, K, N]` weight); pass the last axis for an embedding table
        blocked along its feature dim. `values` may be a numpy array or a
        (possibly nested) Python sequence.
        """
        dtype = AionDType.AION_DTYPE_Q8_0 if dtype is None else normalize_dtype(dtype)
        if dtype not in (AionDType.AION_DTYPE_Q8_0, AionDType.AION_DTYPE_Q4_0):
            raise ValueError(f"quantize expects a quantized dtype, got {dtype}")

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
        dtype: object | None = None,
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
        converter = float if dtype == AionDType.AION_DTYPE_F32 else int
        handle = create_tensor(ctx.ptr, dtype, shp, [converter(value) for value in vals])
        return cls._from_handle(ctx, handle, dtype=dtype, shape=tuple(shp))

    def numel(self) -> int:
        return _elem_count(self.shape)

    def __repr__(self) -> str:
        if getattr(self, "_lazy", False):
            return f"Tensor(lazy, op={self._op!r})"
        if getattr(self, "_closed", True):
            return "Tensor(<closed>)"
        try:
            return f"Tensor(shape={self.shape}, dtype={self.dtype.name})"
        except Exception:
            return "Tensor(<uninitialized>)"

    @property
    def dtype(self) -> AionDType:
        if self._lazy:
            if self._dtype_cache is None:
                self._infer_meta()
            assert self._dtype_cache is not None  # populated by _infer_meta
            return self._dtype_cache
        if self._dtype_cache is None:
            self._dtype_cache = tensor_dtype(self._require_handle())
        return self._dtype_cache

    def _infer_meta(self) -> None:
        """Lower this lazy node once to cache its shape + dtype (core inference)."""
        b, _ = _lower([self])
        b.close()  # throwaway; shapes were cached on the nodes during emit

    @property
    def shape(self) -> tuple[int, ...]:
        if self._lazy:
            if self._shape_cache is None:
                self._infer_meta()
            assert self._shape_cache is not None  # populated by _infer_meta
            return self._shape_cache

        if self._shape_cache is not None:
            return self._shape_cache

        self._shape_cache = tensor_shape(
            self._ctx_owner.ptr, self._require_handle()
        )
        return self._shape_cache

    # --- host conversion and mutation ------------------------------------
    def _flat_values(self) -> list[int | float]:
        self._ensure_realized()
        if _is_quant(self.dtype):
            raise NotImplementedError(
                f"{self.dtype.name}: quantized tensors cannot be converted to Python values"
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

    def copy_from(self, values: object) -> "Tensor":
        """Copy array-like data into this tensor and return ``self``."""
        if self._lazy:
            raise TypeError("cannot mutate a lazy tensor; it is a computed value")
        dt = self.dtype
        if _is_quant(dt):
            raise NotImplementedError(f"{dt.name}: quantized tensors cannot be mutated")

        np = _try_numpy()
        if np is not None:
            from .numpy import _copy_numpy_into_tensor

            _copy_numpy_into_tensor(self, values)
            return self

        if dt == AionDType.AION_DTYPE_F32:
            shape, vals = _flatten_nested(values)
        elif dt == AionDType.AION_DTYPE_I32:
            shape, vals = _flatten_nested_i32(values)
        else:
            raise NotImplementedError(
                f"copy_from without NumPy is not implemented for {dt.name}"
            )
        if shape == () and self.shape == (1,):
            shape = (1,)
        if shape != self.shape:
            raise ValueError(f"shape mismatch: tensor {self.shape} vs data {shape}")
        py = float if dt == AionDType.AION_DTYPE_F32 else int
        write_tensor(
            self._ctx_owner.ptr,
            self._require_handle(),
            [py(cast(Any, value)) for value in vals],
        )
        return self

    def fill(self, value: float) -> "Tensor":
        """In-place fill (f32 only). Returns self."""
        if self.dtype != AionDType.AION_DTYPE_F32:
            raise NotImplementedError("fill() is f32 only")
        write_tensor(
            self._ctx_owner.ptr,
            self._require_handle(),
            [float(value)] * self.numel(),
        )
        return self

    def zero(self) -> "Tensor":
        """In-place zero fill (scalar dtypes). Returns self."""
        dt = self.dtype
        if _is_quant(dt):
            raise NotImplementedError(f"zero() not implemented for dtype={dt.name}")
        n = self.numel()
        # The native façade zero-initializes the temporary C buffer; for f16,
        # uint16 zero is IEEE-754 +0.0.
        zero_tensor(self._ctx_owner.ptr, self._require_handle(), n)
        return self

    # numpy interop (requires numpy; scalar dtypes only).
    def numpy(self) -> NDArray:
        from .numpy import _tensor_to_numpy

        self._ensure_realized()
        return _tensor_to_numpy(self)
