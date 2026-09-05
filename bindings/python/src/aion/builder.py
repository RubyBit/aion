# SPDX-License-Identifier: Apache-2.0
"""PyTorch-like model authoring on top of the Aion C ABI.

Build a graph in code with `Builder`, wiring ops through `TensorRef` handles (which
support `@`, `+`, `-`, `*`, `/`), then either `compile()` it to a runnable
`LoadedModel` in-process or `export()` it to a `.aion` file.

    b = Builder()
    x = b.input((1, 4))
    w = b.param(np.random.randn(4, 3).astype("f4"))
    y = (x @ w).relu()
    model = b.compile({"y": y})

This drives the same Zig `Builder` the core uses, so shape inference, op
validation, and serialization all come from one source of truth.
"""
from __future__ import annotations

import contextlib
from types import TracebackType
from collections.abc import Generator
from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Tuple, Union, overload

from .context import Context
from .device import DeviceLike, _normalize_device
from .dtype import dtype_name, float32, normalize_dtype as _as_dtype
from .errors import AionError
from ._ffi.authoring import (
    builder_topk,
    AttentionAttrs,
    AxisAttrs,
    CastAttrs,
    Conv1DAttrs,
    Conv2DAttrs,
    ElemwiseAttrs,
    GatherAttrs,
    MatmulAttrs,
    NormAttrs,
    OpAttrs,
    OptionalAxisAttrs,
    ReduceAttrs,
    RegionId,
    RelposMhaAttrs,
    ReshapeAttrs,
    Rope1DAttrs,
    SliceAttrs,
    StftAttrs,
    UnaryAttrs,
    ValueId,
    ViewDims,
    builder_add_dim_symbol,
    builder_add_input_role,
    builder_add_metadata,
    builder_add_output_alias,
    builder_begin_region,
    builder_end_region,
    builder_if,
    builder_input,
    builder_loop,
    builder_clear_outputs,
    builder_mark_output,
    builder_begin_auto_scope,
    builder_begin_scope,
    builder_constant,
    builder_end_scope,
    builder_filled_vec,
    builder_has_param_named,
    builder_name,
    builder_param,
    builder_param_kind,
    builder_param_named,
    builder_scope_path,
    builder_value_name,
    builder_value_dtype,
    builder_symbol_size,
    builder_value_dim_symbol,
    builder_value_shape,
    compile_builder,
    create_builder,
    destroy_builder,
    emit_op,
    export_builder,
)
from ._ffi.handles import BuilderHandle
from .enums import (
    AionBinaryOp,
    AionDType,
    AionInputRoleKind,
    AionOp,
    AionPadMode,
    AionReduceOp,
    AionUnaryOp,
)
from .tensor import Tensor
from .types import ArrayLike, AttentionWindow, DTypeLike, NDArray, Shape

if TYPE_CHECKING:
    from .model import LoadedModel

# A weight source: an existing Tensor, a numpy array, or a nested Python list.
type WeightData = Tensor | ArrayLike
# Which input axes are dynamic (vary at runtime): a sequence of axis indices
# (auto-named symbols) or a {axis: symbol_name} mapping (reuse a name to tie axes
# across inputs to the same runtime size). The declared int at each axis is the
# authoring placeholder.
DynamicAxes = Union[None, Sequence[int], Mapping[int, str]]
# What `compile`/`export` accept as outputs: one value, a list, or {name: value}.
OutputsLike = Union["TensorRef", Sequence["TensorRef"], Mapping[str, "TensorRef"]]
_PAD_MODES = {
    "zero": AionPadMode.AION_PAD_ZERO,
    "reflect": AionPadMode.AION_PAD_REFLECT,
}

class TensorRef:
    """A handle on a value in one `Builder`'s graph (an opaque u32 id).

    This is *the* authoring handle — what `Builder` ops and `nn` layers take and
    return. It carries the operators and fluent forms (`(x @ w).relu()`) plus
    authoring-time `shape`/`dtype` introspection via eager per-op inference, and
    it knows its own builder, so a layer needs no ambient state.

    It is the counterpart of `TensorRef` in the Zig API; `aion.Tensor` is the
    separate *data* handle, the same split Zig makes with `api.Tensor`.
    """

    __slots__ = ("_b", "id", "_name")

    def __init__(self, builder: "Builder", vid: ValueId | int):
        self._b = builder
        self.id: ValueId = ValueId(int(vid))
        self._name = None

    # --- operators ---------------------------------------------------------
    def __matmul__(self, other: "TensorRef") -> "TensorRef":
        return self._b.matmul(self, other)

    def __add__(self, other: "TensorRef") -> "TensorRef":
        return self._b.add(self, other)

    def __sub__(self, other: "TensorRef") -> "TensorRef":
        return self._b.sub(self, other)

    def __mul__(self, other: "TensorRef") -> "TensorRef":
        return self._b.mul(self, other)

    def __truediv__(self, other: "TensorRef") -> "TensorRef":
        return self._b.div(self, other)

    # --- fluent method forms ----------------------------------------------
    def relu(self) -> "TensorRef":
        return self._b.unary("relu", self)

    def gelu(self) -> "TensorRef":
        return self._b.unary("gelu", self)

    def silu(self) -> "TensorRef":
        return self._b.unary("silu", self)

    def sigmoid(self) -> "TensorRef":
        return self._b.unary("sigmoid", self)

    def tanh(self) -> "TensorRef":
        return self._b.unary("tanh", self)

    def softmax(self, axis: int = -1) -> "TensorRef":
        return self._b.softmax(self, axis)

    def reshape(self, shape: Sequence[Union[int, str]]) -> "TensorRef":
        # Same vocabulary as `Builder.reshape`: a str names a declared dim symbol, so
        # a symbolic axis survives the fluent form too.
        return self._b.reshape(self, shape)

    def transpose2d(self) -> "TensorRef":
        return self._b.transpose2d(self)

    def cast(self, dtype: DTypeLike) -> "TensorRef":
        return self._b.cast(self, dtype)

    def rename(self, name: str) -> "TensorRef":
        self._b.name(self, name)
        self._name = name
        return self

    # --- authoring-time introspection -------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        """Shape at the authoring placeholder sizes (eager per-op inference).

        Concrete ints throughout — a dynamic axis reports the placeholder it was
        declared with, propagated to derived values. The compiled/exported model
        still serves any size on dynamic axes.
        """
        return builder_value_shape(
            self._b._ctx_owner.ptr, self._b.ptr, self.id
        )

    @property
    def dims(self) -> tuple[Union[int, str], ...]:
        """Shape as *authored*: a str on a free axis, an int on a fixed one.

        A free axis is one some input declared with `symbolic_dim`; inference
        carries the symbol to every value derived from it. Feed these straight back
        to `reshape`/`slice`, which speak the same vocabulary, and an axis the caller
        made free stays free instead of freezing at its placeholder size:

            q = (x @ w).reshape((*x.dims[:2], heads, head_dim))

        `shape` is the same thing with the placeholder sizes substituted in.
        """
        sizes = self.shape
        return tuple(
            builder_value_dim_symbol(
                self._b._ctx_owner.ptr, self._b.ptr, self.id, axis
            )
            or size
            for axis, size in enumerate(sizes)
        )

    @property
    def dtype(self) -> AionDType:
        return builder_value_dtype(
            self._b._ctx_owner.ptr, self._b.ptr, self.id
        )

    @property
    def ndim(self) -> int:
        return len(self.shape)

    # --- evaluation --------------------------------------------------------

    def eval(self) -> "Tensor":
        """Compute this value and return the result as data.

        For looking at a value while authoring — `print(y)`, `y.numpy()` — without
        naming outputs or building a model. `compile` hands the model a *copy* of
        the graph, so this leaves the builder untouched and you keep building.
        Compilation is pruned to this value's cone, so the cost is what the value
        needs and not what the whole graph holds.

        Memoized: the graph is append-only, so a value's result cannot change.
        Raises if the value depends on an input with no data bound — that is not a
        computable value, and the message names the input.
        """
        cached = self._b._eval_cache.get(self.id)
        if cached is not None:
            return cached

        name = "__eval__"
        model = self._b.compile({name: self})
        try:
            # Compilation is pruned to this value's cone, so any public input the
            # model reports is one this value genuinely depends on. Refuse rather
            # than run: the runtime would zero-fill them and hand back numbers that
            # look real.
            needed = model.input_names()
            if needed:
                raise ValueError(
                    "cannot evaluate: depends on input(s) with no data bound: "
                    + ", ".join(needed)
                    + ". Bind them and run a compiled model instead."
                )
            model.run()
            # Copy out of the model's storage so the result outlives it.
            result = model.output_tensor(name).tolist()
        finally:
            model.close()

        from .tensor import Tensor

        value = Tensor(result, ctx=self._b._ctx_owner)
        self._b._eval_cache[self.id] = value
        return value

    def numpy(self) -> "NDArray":
        """Evaluate and return the result as a numpy array."""
        return self.eval().numpy()

    def item(self) -> float | int:
        """Evaluate and return a one-element result as a Python scalar."""
        return self.eval().item()

    def tolist(self) -> object:
        """Evaluate and return the result as nested Python lists."""
        return self.eval().tolist()

    def __repr__(self) -> str:
        try:
            shape, dtype = self.shape, dtype_name(self.dtype)
        except Exception:
            return f"TensorRef(id={self.id})"
        try:
            return f"TensorRef(shape={shape}, dtype={dtype}, value={self.tolist()!r})"
        except Exception:
            # Not computable (e.g. depends on an unbound input) — still describe it.
            return f"TensorRef(shape={shape}, dtype={dtype})"


class Builder:
    """Authors a graph, then compiles or exports it."""

    def __init__(self, ctx: "Context | None" = None) -> None:
        from .context import get_default_context

        if ctx is None:
            ctx = get_default_context()
        self._ctx_owner = ctx
        self._b: BuilderHandle | None = create_builder(ctx.ptr)
        self._closed = False
        # Keep param tensors alive for the builder's lifetime.
        self._params: list[Tensor] = []
        self._symbol_counter = 0
        # value id -> evaluated result. Safe to memoize: the graph is append-only,
        # so what a value computes to cannot change once computed.
        self._eval_cache: dict[int, Tensor] = {}
        ctx._register_child(self)

    @property
    def ptr(self) -> BuilderHandle:
        return self._require_handle()

    def _require_handle(self) -> BuilderHandle:
        if self._b is None:
            raise RuntimeError("Builder is closed")
        return self._b

    @property
    def context(self) -> "Context":
        return self._ctx_owner

    def close(self) -> None:
        if self._closed:
            return
        handle = self._b
        if handle is not None:
            destroy_builder(handle)
        try:
            self._ctx_owner._unregister_child(self)
        except Exception:
            pass
        self._b = None
        self._closed = True
        self._params.clear()

    def __enter__(self) -> "Builder":
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
                handle = getattr(self, "_b", None)
                if handle is not None:
                    destroy_builder(handle)
        except Exception:
            pass

    # --- inputs & weights --------------------------------------------------
    def input(
        self,
        shape: Shape,
        *,
        dtype: DTypeLike = float32,
        dynamic: DynamicAxes = None,
    ) -> TensorRef:
        """Declare a public runtime input.

        `shape` is all concrete ints. `dynamic` marks which axes vary at runtime
        (the model then serves any size on them); the declared int at a dynamic
        axis is its authoring **placeholder** — the size eager inference and
        `.shape` use, propagated to derived values. `dynamic` is either a
        sequence of axis indices (auto-named symbols) or a `{axis: name}` mapping
        (reuse a name across inputs to tie their axes to the same runtime size).
        """
        dt = _as_dtype(dtype)
        dims = [int(d) for d in shape]
        symbolic = _resolve_dynamic(dynamic, len(dims), self)

        value_id = builder_input(self._ctx_owner.ptr, self.ptr, dt, dims)
        v = TensorRef(self, value_id)
        for axis, name in symbolic:
            self._add_dim_symbol(v, axis, name)
        return v

    def _auto_symbol_name(self) -> str:
        name = f"sym{self._symbol_counter}"
        self._symbol_counter += 1
        return name

    def _add_dim_symbol(self, value: TensorRef, axis: int, name: str) -> None:
        builder_add_dim_symbol(
            self._ctx_owner.ptr, self.ptr, value.id, axis, name
        )

    def param(
        self,
        data: WeightData,
        *,
        dtype: DTypeLike = float32,
        shape: Optional[Shape] = None,
        quant_axis: Optional[int] = None,
    ) -> TensorRef:
        """Bind a weight from data (a `Tensor`, numpy array, or nested list).

        A float `dtype` binds the data directly; a quantized `dtype`
        (``aion.q8_0``) quantizes it in the core first, blocking along `quant_axis`
        (default: the matmul-B reduction axis, rank-2; pass the last axis for an
        embedding table). When `data` is already a `Tensor`, `dtype`/`shape`/
        `quant_axis` are ignored.
        """
        t = self._as_param_tensor(data, dtype=dtype, shape=shape, quant_axis=quant_axis)
        self._params.append(t)
        return TensorRef(self, builder_param(self._ctx_owner.ptr, self.ptr, t.ptr))

    def _as_param_tensor(
        self,
        data: WeightData,
        *,
        dtype: DTypeLike = float32,
        shape: Optional[Shape] = None,
        quant_axis: Optional[int] = None,
    ) -> Tensor:
        """Coerce weight data to a Tensor this context owns (see `param`)."""
        if isinstance(data, Tensor):
            return data
        dt = _as_dtype(dtype)
        if dt in (AionDType.AION_DTYPE_Q8_0, AionDType.AION_DTYPE_Q4_0):
            if shape is None:
                shape = _infer_shape(data)
            return Tensor.quantize(self._ctx_owner, shape, data, dtype=dt, quant_axis=quant_axis)
        return Tensor(data, ctx=self._ctx_owner, dtype=dt)

    def name(self, value: TensorRef, name: str) -> TensorRef:
        builder_name(self._ctx_owner.ptr, self.ptr, value.id, name)
        return value

    def param_named(
        self,
        data: WeightData,
        name: str,
        *,
        dtype: DTypeLike = float32,
        shape: Optional[Shape] = None,
        quant_axis: Optional[int] = None,
    ) -> TensorRef:
        """Bind a weight under a semantic, scope-qualified name (`.../weight`).

        The name persists into the package as a `debug_name`, which is the key the
        load- and swap-by-name paths use. Prefer this over `param`, whose generated
        name is positional and shifts if construction order changes.
        """
        t = self._as_param_tensor(data, dtype=dtype, shape=shape, quant_axis=quant_axis)
        self._params.append(t)
        return TensorRef(self, builder_param_named(self._ctx_owner.ptr, self.ptr, t.ptr, name))

    # --- scopes ------------------------------------------------------------

    @property
    def scope_path(self) -> str:
        """The `/`-joined path of all open scopes (`""` when none are open)."""
        return builder_scope_path(self.ptr)

    @contextlib.contextmanager
    def scope(self, name: str) -> Generator[str, None, None]:
        """Open a named scope, nested inside any already-open one.

        Parameters bound inside land under `<path>/<param>`, which is what makes
        them addressable in a loaded model.
        """
        depth = builder_begin_scope(self._ctx_owner.ptr, self.ptr, name)
        try:
            yield name
        finally:
            builder_end_scope(self._ctx_owner.ptr, self.ptr, depth)

    @contextlib.contextmanager
    def auto_scope(self, base: str) -> Generator[str, None, None]:
        """Open a scope named `{base}#{n}`, numbered per parent scope.

        Note the tradeoff: numbering follows bind order, so inserting a layer ahead
        of others shifts their numbers and re-keys their parameters. Use `scope`
        with an explicit name for anything you intend to look up later.
        """
        depth, segment = builder_begin_auto_scope(self._ctx_owner.ptr, self.ptr, base)
        try:
            yield segment
        finally:
            builder_end_scope(self._ctx_owner.ptr, self.ptr, depth)

    # --- synthesized constants ---------------------------------------------

    def constant(self, value: float) -> TensorRef:
        """A cached one-element f32 constant for elementwise scalar broadcasting.

        Aion has no scalar-typed operand, so `x * k` uses a size-one tensor. This
        keeps the operand one element regardless of what it multiplies and shares
        it across every use of the same value.
        """
        return TensorRef(self, builder_constant(self._ctx_owner.ptr, self.ptr, value))

    def zeros(self, dim: int) -> TensorRef:
        """A cached `[dim]` f32 zeros vector: the identity `beta` of a norm."""
        return TensorRef(self, builder_filled_vec(self._ctx_owner.ptr, self.ptr, dim, 0.0))

    def ones(self, dim: int) -> TensorRef:
        """A cached `[dim]` f32 ones vector: the identity `gamma` of a norm."""
        return TensorRef(self, builder_filled_vec(self._ctx_owner.ptr, self.ptr, dim, 1.0))

    # --- parameter introspection -------------------------------------------

    def param_kind(self, value: TensorRef) -> Optional[str]:
        """`"user"`, `"synthesized"`, or None if `value` is not a bound parameter.

        Distinguishes a model weight from a constant the Builder had to invent for
        an op that requires an operand where the maths wants a number.
        """
        kind = builder_param_kind(self.ptr, value.id)
        return {1: "user", 2: "synthesized"}.get(kind)

    def has_param_named(self, name: str) -> bool:
        """Whether this graph binds a parameter under `name`, of either kind."""
        return builder_has_param_named(self.ptr, name)

    def value_name(self, value: TensorRef) -> str:
        """The debug name attached to `value`, or "" when it has none.

        This is the name that lands in the package, so it is the authority on
        where a parameter lives — nothing needs to recompute the path.
        """
        return builder_value_name(self.ptr, value.id)

    # --- op emit -----------------------------------------------------------
    def _emit(
        self,
        op: AionOp,
        inputs: Sequence[TensorRef],
        attrs: OpAttrs = None,
    ) -> TensorRef:
        value_id = emit_op(
            self._ctx_owner.ptr,
            self.ptr,
            op,
            [value.id for value in inputs],
            attrs,
        )
        return TensorRef(self, value_id)

    # Structured ops.
    def matmul(self, a: TensorRef, b: TensorRef, alpha: float = 1.0, beta: float = 0.0) -> TensorRef:
        return self._emit(
            AionOp.AION_OP_MATMUL, (a, b), MatmulAttrs(alpha, beta)
        )

    def matmul_nt(self, a: TensorRef, b: TensorRef, alpha: float = 1.0, beta: float = 0.0) -> TensorRef:
        return self._emit(
            AionOp.AION_OP_MATMUL_NT, (a, b), MatmulAttrs(alpha, beta)
        )

    def _elemwise(self, op: AionBinaryOp, a: TensorRef, b: TensorRef) -> TensorRef:
        return self._emit(
            AionOp.AION_OP_ELEMWISE, (a, b), ElemwiseAttrs(int(op))
        )

    def add(self, a: TensorRef, b: TensorRef) -> TensorRef:
        return self._elemwise(AionBinaryOp.AION_BINARY_ADD, a, b)

    def sub(self, a: TensorRef, b: TensorRef) -> TensorRef:
        return self._elemwise(AionBinaryOp.AION_BINARY_SUB, a, b)

    def mul(self, a: TensorRef, b: TensorRef) -> TensorRef:
        return self._elemwise(AionBinaryOp.AION_BINARY_MUL, a, b)

    def div(self, a: TensorRef, b: TensorRef) -> TensorRef:
        return self._elemwise(AionBinaryOp.AION_BINARY_DIV, a, b)

    _UNARY = {
        "relu": AionUnaryOp.AION_UNARY_RELU,
        "gelu": AionUnaryOp.AION_UNARY_GELU,
        "silu": AionUnaryOp.AION_UNARY_SILU,
        "sigmoid": AionUnaryOp.AION_UNARY_SIGMOID,
        "tanh": AionUnaryOp.AION_UNARY_TANH,
        "sqrt": AionUnaryOp.AION_UNARY_SQRT,
        "log": AionUnaryOp.AION_UNARY_LOG,
    }

    def unary(self, op: str, a: TensorRef) -> TensorRef:
        u = self._UNARY[op]
        return self._emit(AionOp.AION_OP_UNARY, (a,), UnaryAttrs(int(u)))

    def softmax(self, a: TensorRef, axis: int = -1) -> TensorRef:
        return self._emit(AionOp.AION_OP_SOFTMAX, (a,), AxisAttrs(axis))

    def rmsnorm(self, x: TensorRef, gamma: TensorRef, beta: TensorRef, *, eps: float = 1e-6, normalized_shape: Sequence[int]) -> TensorRef:
        return self._norm(AionOp.AION_OP_RMSNORM, x, gamma, beta, eps, normalized_shape)

    def layernorm(self, x: TensorRef, gamma: TensorRef, beta: TensorRef, *, eps: float = 1e-5, normalized_shape: Sequence[int]) -> TensorRef:
        return self._norm(AionOp.AION_OP_LAYERNORM, x, gamma, beta, eps, normalized_shape)

    def _norm(
        self,
        op: AionOp,
        x: TensorRef,
        gamma: TensorRef,
        beta: TensorRef,
        eps: float,
        normalized_shape: Sequence[int],
    ) -> TensorRef:
        attrs = NormAttrs(
            float(eps), tuple(int(dim) for dim in normalized_shape)
        )
        return self._emit(op, (x, gamma, beta), attrs)

    def gather(
        self,
        data: TensorRef,
        indices: TensorRef,
        *,
        axis: int = 0,
        batch_dims: int = 0,
    ) -> TensorRef:
        """Gather slices along an axis, sharing leading batch dimensions."""
        return self._emit(
            AionOp.AION_OP_GATHER,
            (data, indices),
            GatherAttrs(axis, batch_dims),
        )

    def dim(self, value: TensorRef, axis: int) -> TensorRef:
        """Reify one concrete runtime dimension as an i32 one-element tensor."""
        return self._emit(AionOp.AION_OP_DIM, (value,), AxisAttrs(axis))

    def iota(self, shape_like: TensorRef, axis: int) -> TensorRef:
        """Return i32 coordinates with `shape_like`'s shape along `axis`."""
        return self._emit(AionOp.AION_OP_IOTA, (shape_like,), AxisAttrs(axis))

    def rope1d(self, x: TensorRef, positions: TensorRef, *, base_frequency: float, scale_factor: float = 1.0, rope_proportion: float = 1.0) -> TensorRef:
        attrs = Rope1DAttrs(base_frequency, scale_factor, rope_proportion)
        return self._emit(AionOp.AION_OP_ROPE1D, (x, positions), attrs)

    def concat(self, values: Sequence[TensorRef], axis: int) -> TensorRef:
        return self._emit(
            AionOp.AION_OP_CONCAT, tuple(values), AxisAttrs(axis)
        )

    def _resolve_view_dims(
        self, dims: Sequence[Union[int, str]]
    ) -> ViewDims:
        """Resolve a view-op dim list (ints and dim-symbol names) to a concrete
        placeholder `size_t[]` plus an optional `char*[]` of per-axis symbol names
        (NULL where constant). Returns (concrete_cdata, symbols_cdata_or_NULL,
        keepalive) — the keepalive holds the individual C strings alive."""
        concrete: list[int] = []
        names: list[Optional[str]] = []
        for d in dims:
            if isinstance(d, str):
                # The placeholder lives with the declaration, on the builder itself,
                # so it is recorded once rather than mirrored on this side.
                try:
                    concrete.append(
                        builder_symbol_size(self._ctx_owner.ptr, self.ptr, d)
                    )
                except AionError as exc:
                    raise ValueError(
                        f"unknown dim symbol {d!r}; declare it on an input via "
                        f"input(..., dynamic={{axis: {d!r}}})") from exc
                names.append(d)
            else:
                concrete.append(int(d))
                names.append(None)
        return ViewDims(tuple(concrete), tuple(names))

    def reshape(self, a: TensorRef, shape: Sequence[Union[int, str]]) -> TensorRef:
        return self._emit(
            AionOp.AION_OP_RESHAPE,
            (a,),
            ReshapeAttrs(self._resolve_view_dims(shape)),
        )

    def transpose2d(self, a: TensorRef) -> TensorRef:
        return self._emit(AionOp.AION_OP_TRANSPOSE2D, (a,))

    def cast(self, a: TensorRef, dtype: DTypeLike) -> TensorRef:
        dt = _as_dtype(dtype)
        return self._emit(AionOp.AION_OP_CAST, (a,), CastAttrs(dt))

    def argmax(self, a: TensorRef, axis: int = -1) -> TensorRef:
        return self._emit(AionOp.AION_OP_ARGMAX, (a,), AxisAttrs(axis))

    def topk(
        self,
        a: TensorRef,
        k: int,
        axis: int = -1,
        *,
        largest: bool = True,
    ) -> Tuple[TensorRef, TensorRef]:
        """`(values, indices)`: the `k` best entries along `axis` and where they came from.

        Sorted best-first, ties going to the lowest index, so the result is a
        function of the input alone. `largest=False` gives the k smallest. Both
        outputs have the input's shape with `axis` resized to `k`; the indices are
        int32. Lowering implements the last axis -- transpose to reach another.
        """
        values, indices = builder_topk(
            self._ctx_owner.ptr, self.ptr, a.id, int(k), int(axis), bool(largest)
        )
        return TensorRef(self, values), TensorRef(self, indices)

    def reduce(self, op: str, a: TensorRef, axis: Optional[int] = None) -> TensorRef:
        r = AionReduceOp.AION_REDUCE_SUM if op == "sum" else AionReduceOp.AION_REDUCE_MEAN

        return self._emit(
            AionOp.AION_OP_REDUCE, (a,), ReduceAttrs(int(r), axis)
        )

    # --- comparisons (i32 {0,1} output) -----------------------------------
    _COMPARE = {
        "eq": AionBinaryOp.AION_BINARY_EQ,
        "ne": AionBinaryOp.AION_BINARY_NE,
        "lt": AionBinaryOp.AION_BINARY_LT,
        "gt": AionBinaryOp.AION_BINARY_GT,
        "le": AionBinaryOp.AION_BINARY_LE,
        "ge": AionBinaryOp.AION_BINARY_GE,
    }

    def compare(self, op: str, a: TensorRef, b: TensorRef) -> TensorRef:
        """Elementwise comparison (`eq/ne/lt/gt/le/ge`) producing i32 {0,1}."""
        return self._elemwise(self._COMPARE[op], a, b)

    # --- convolutions -----------------------------------------------------
    def conv1d(self, x: TensorRef, weight: TensorRef, bias: Optional[TensorRef] = None, *,
               stride: int = 1, dilation: int = 1, pad_left: int = 0, pad_right: int = 0,
               groups: int = 1, pad_mode: str = "zero") -> TensorRef:
        inputs = (x, weight) if bias is None else (x, weight, bias)

        attrs = Conv1DAttrs(
            stride,
            dilation,
            pad_left,
            pad_right,
            groups,
            int(_PAD_MODES[pad_mode]),
        )
        return self._emit(AionOp.AION_OP_CONV1D, inputs, attrs)

    def conv2d(self, x: TensorRef, weight: TensorRef, bias: Optional[TensorRef] = None, *,
               stride_h: int = 1, stride_w: int = 1, dilation_h: int = 1, dilation_w: int = 1,
               pad_top: int = 0, pad_bottom: int = 0, pad_left: int = 0, pad_right: int = 0,
               groups: int = 1, pad_mode: str = "zero") -> TensorRef:
        inputs = (x, weight) if bias is None else (x, weight, bias)

        attrs = Conv2DAttrs(
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            groups,
            int(_PAD_MODES[pad_mode]),
        )
        return self._emit(AionOp.AION_OP_CONV2D, inputs, attrs)

    # --- signal (FFT) -----------------------------------------------------
    def stft(self, signal: TensorRef, window: TensorRef, *, n_fft: int, hop_length: int,
             center: bool = False) -> TensorRef:
        return self._emit(
            AionOp.AION_OP_STFT,
            (signal, window),
            StftAttrs(n_fft, hop_length, center),
        )

    def rfft(self, x: TensorRef) -> TensorRef:
        return self._emit(AionOp.AION_OP_RFFT, (x,))

    # --- views ------------------------------------------------------------
    def squeeze(self, a: TensorRef, axis: Optional[int] = None) -> TensorRef:
        return self._emit(
            AionOp.AION_OP_SQUEEZE, (a,), OptionalAxisAttrs(axis)
        )

    def unsqueeze(self, a: TensorRef, axis: int) -> TensorRef:
        return self._emit(
            AionOp.AION_OP_UNSQUEEZE, (a,), AxisAttrs(axis)
        )

    def slice(self, a: TensorRef, starts: Sequence[int], lens: Sequence[Union[int, str]]) -> TensorRef:
        """N-D slice (one `start`/`len` per axis). A `len` may be an int (constant)
        or a dim-symbol name (declared via `input(dynamic=...)`) for a symbolic axis."""
        starts_l = [int(s) for s in starts]
        if len(starts_l) != len(lens):
            raise ValueError("slice starts and lens must have equal length")
        attrs = SliceAttrs(
            tuple(starts_l), self._resolve_view_dims(lens)
        )
        return self._emit(AionOp.AION_OP_SLICE, (a,), attrs)

    def slice_last_dim(self, a: TensorRef, start: int, length: int) -> TensorRef:
        """Take `length` elements from `start` along the last axis, keeping every
        leading axis whole. Mirrors `Builder.sliceLastDim` in the Zig API.

        This is how a fused projection is split back into its parts (QKV,
        gate/up), which otherwise means spelling out full starts/lens per call.
        """
        dims = a.dims
        if not dims:
            raise ValueError("slice_last_dim needs a ranked value")
        starts = [0] * len(dims)
        lens = list(dims)
        starts[-1] = int(start)
        lens[-1] = int(length)
        return self.slice(a, starts, lens)

    # --- attention --------------------------------------------------------
    def attention(self, q: TensorRef, k: TensorRef, v: TensorRef, *,
                  query_positions: Optional[TensorRef] = None,
                  kv_lengths: Optional[TensorRef] = None, scale: float,
                  window: AttentionWindow = AttentionWindow.CAUSAL,
                  attn_logits_soft_cap: float = 0.0) -> TensorRef:
        """Grouped-query attention in canonical time-major layout.

        Q is `[batch, query, heads, key_dim]`; K/V are
        `[batch, key, kv_heads, key/value_dim]`. `query_positions` and
        `kv_lengths` are independent; they default to row indices and the full key
        length. Causal unequal-length attention requires explicit positions.
        """
        inputs = [q, k, v]
        if query_positions is not None:
            inputs.append(query_positions)
        if kv_lengths is not None:
            inputs.append(kv_lengths)
        return self._emit(
            AionOp.AION_OP_ATTENTION,
            tuple(inputs),
            AttentionAttrs(
                scale, window, attn_logits_soft_cap,
                query_positions is not None, kv_lengths is not None,
            ),
        )

    def relpos_mha(self, q: TensorRef, k: TensorRef, v: TensorRef, pos_emb: TensorRef, bu: TensorRef, bv: TensorRef,
                   mask: Optional[TensorRef] = None, *, scale: float,
                   window: AttentionWindow = AttentionWindow.FULL,
                   relative_zero_index: int = 0,
                   attn_logits_soft_cap: float = 0.0) -> TensorRef:
        """Relative-position MHA. The head count is `q`'s dim 2 — never passed.
        `window` is structural, so the kernels skip keys outside it; keep `mask` for
        what an interval cannot express, such as streaming padding."""
        inputs: list[TensorRef] = [q, k, v, pos_emb, bu, bv]
        if mask is not None:
            inputs.append(mask)
        return self._emit(
            AionOp.AION_OP_RELPOS_MHA,
            inputs,
            RelposMhaAttrs(scale, window, relative_zero_index, attn_logits_soft_cap),
        )

    # --- recurrent / indexed / misc ---------------------------------------
    def lstm_cell(self, x: TensorRef, h: TensorRef, c: TensorRef, w_ih: TensorRef, w_hh: TensorRef,
                  b_ih: Optional[TensorRef] = None, b_hh: Optional[TensorRef] = None) -> TensorRef:
        """One LSTM step -> `[batch, 2*hidden]` = (h_t | c_t). Biases are both-or-neither."""
        if (b_ih is None) != (b_hh is None):
            raise ValueError("lstm_cell requires both b_ih and b_hh or neither")
        inputs: list[TensorRef] = [x, h, c, w_ih, w_hh]
        if b_ih is not None and b_hh is not None:
            inputs.extend((b_ih, b_hh))
        return self._emit(AionOp.AION_OP_LSTM_CELL, inputs)

    def sequence_append(self, cache: TensorRef, new_kv: TensorRef, end_index: TensorRef) -> TensorRef:
        return self._emit(AionOp.AION_OP_SEQUENCE_APPEND, (cache, new_kv, end_index))

    def scatter_row(self, buf: TensorRef, index: TensorRef, src: TensorRef) -> TensorRef:
        return self._emit(AionOp.AION_OP_SCATTER_ROW, (buf, index, src))

    def copy(self, a: TensorRef) -> TensorRef:
        return self._emit(AionOp.AION_OP_COPY, (a,))

    # --- output aliases & input roles -------------------------------------
    def add_output_alias(self, input_value: TensorRef, output_value: TensorRef) -> None:
        """Alias a graph input to an output (io-aliased recurrent state write-back).

        The output must already be marked (see `mark_output`)."""
        builder_add_output_alias(
            self._ctx_owner.ptr,
            self.ptr,
            input_value.id,
            output_value.id,
        )

    def add_input_role(self, value: TensorRef, kind: AionInputRoleKind, *,
                       axis: Optional[int] = None, capacity_symbol: Optional[str] = None,
                       zero_init: bool = True, growable: bool = False,
                       retained_history_tokens: int = 0) -> None:
        """Tag an input with a runtime role (KV cache, positions, tokens, state, …).

        `capacity_symbol` names a dim symbol declared on `value` (via
        `input(dynamic={axis: name})`) whose size the runtime supplies at load."""
        builder_add_input_role(
            self._ctx_owner.ptr,
            self.ptr,
            value.id,
            int(kind),
            axis=axis,
            capacity_symbol=capacity_symbol,
            zero_init=zero_init,
            growable=growable,
            retained_history_tokens=retained_history_tokens,
        )

    # --- control flow (regions) -------------------------------------------
    def begin_region(self) -> None:
        """Start a region (an `if` branch body or `loop` body). No nesting."""
        builder_begin_region(self._ctx_owner.ptr, self.ptr)

    def end_region(self, outputs: Sequence[TensorRef]) -> RegionId:
        """Close the active region; returns a region id for `if_`/`loop`."""
        return builder_end_region(
            self._ctx_owner.ptr,
            self.ptr,
            [value.id for value in outputs],
        )

    def if_(
        self, cond: TensorRef, then_region: RegionId, else_region: RegionId
    ) -> TensorRef:
        """Single-output conditional over an i32 `[1]` predicate `cond`."""
        value_id = builder_if(
            self._ctx_owner.ptr,
            self.ptr,
            cond.id,
            then_region,
            else_region,
        )
        return TensorRef(self, value_id)

    @overload
    def loop(self, carried: TensorRef, body_region: RegionId, trip: int, *,
             cond_carry: Optional[int] = None, check_before: bool = True) -> TensorRef: ...
    @overload
    def loop(self, carried: Sequence[TensorRef], body_region: RegionId, trip: int, *,
             cond_carry: Optional[int] = None, check_before: bool = True) -> list[TensorRef]: ...

    def loop(
        self,
        carried: Union[TensorRef, Sequence[TensorRef]],
        body_region: RegionId,
        trip: int,
        *,
        cond_carry: Optional[int] = None,
        check_before: bool = True,
    ) -> Union[TensorRef, list[TensorRef]]:
        """Run `body_region` up to `trip` times, threading the carried value(s).

        Pass a single `TensorRef` (returns a `TensorRef`) or a sequence for a multi-carry
        loop (returns a `list`). `cond_carry` names the carry index holding the i32
        `[1]` continue predicate, if any.
        """
        single = isinstance(carried, TensorRef)
        inits = [carried] if single else list(carried)
        output_ids = builder_loop(
            self._ctx_owner.ptr,
            self.ptr,
            [value.id for value in inits],
            body_region,
            trip,
            cond_carry=cond_carry,
            check_before=check_before,
        )
        values = [TensorRef(self, value_id) for value_id in output_ids]
        return values[0] if single else values

    # --- declarations & terminals -----------------------------------------
    def mark_output(self, value: TensorRef, name: str) -> None:
        builder_mark_output(
            self._ctx_owner.ptr, self.ptr, value.id, name
        )

    def add_metadata(self, key: str, value: str) -> None:
        builder_add_metadata(self._ctx_owner.ptr, self.ptr, key, value)

    def clear_outputs(self) -> None:
        """Forget every output marked so far."""
        builder_clear_outputs(self._ctx_owner.ptr, self.ptr)

    def _mark_outputs(self, outputs: OutputsLike) -> None:
        # Replace rather than extend: `compile(outputs)` means *these* outputs.
        # `mark_output` accumulates on the builder, so without clearing, a second
        # compile would silently carry the first one's outputs.
        self.clear_outputs()

        # A `{name: value}` mapping names outputs explicitly; otherwise a value's
        # own `.rename(...)` (its `_name`) is the default, falling back to the
        # positional `output{i}`.
        if isinstance(outputs, TensorRef):
            self.mark_output(outputs, getattr(outputs, "_name", None) or "output0")
        elif isinstance(outputs, Mapping):
            for name, v in outputs.items():
                self.mark_output(v, str(name))
        else:
            for i, v in enumerate(outputs):
                self.mark_output(v, getattr(v, "_name", None) or f"output{i}")

    def compile(self, outputs: Optional[OutputsLike] = None, *, device: DeviceLike = None) -> "LoadedModel":
        """Compile to an in-process `LoadedModel` (concrete shapes only)."""
        from .model import LoadedModel

        if outputs is not None:
            self._mark_outputs(outputs)
        kind, index = _normalize_device(device)
        handle = compile_builder(
            self._ctx_owner.ptr, self.ptr, int(kind), int(index)
        )
        return LoadedModel(self._ctx_owner, handle)

    def export(self, path: str, outputs: Optional[OutputsLike] = None) -> None:
        """Serialize the graph to a `.aion` file at `path`."""
        if outputs is not None:
            self._mark_outputs(outputs)
        export_builder(self._ctx_owner.ptr, self.ptr, str(path))


def _resolve_dynamic(dynamic: DynamicAxes, rank: int, b: "Builder") -> list[tuple[int, str]]:
    """Normalize `dynamic` into a list of (axis, symbol_name), auto-naming axes
    given as bare indices."""
    if dynamic is None:
        return []
    out: list[tuple[int, str]] = []
    if isinstance(dynamic, Mapping):
        items = [(int(ax), str(name)) for ax, name in dynamic.items()]
    else:
        items = [(int(ax), b._auto_symbol_name()) for ax in dynamic]
    for axis, name in items:
        if axis < 0 or axis >= rank:
            raise ValueError(f"dynamic axis {axis} out of range for rank {rank}")
        out.append((axis, name))
    return out


def _infer_shape(data: WeightData) -> tuple[int, ...]:
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None and isinstance(data, np.ndarray):
        return tuple(int(x) for x in data.shape)
    from .tensor import _flatten_nested

    shape, _ = _flatten_nested(data)
    return shape
