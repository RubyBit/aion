# SPDX-License-Identifier: Apache-2.0
"""PyTorch-like model authoring on top of the Aion C ABI.

Build a graph in code with `Builder`, wiring ops through `Value` handles (which
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

from types import TracebackType
from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Union, overload

from .context import Context
from .device import DeviceLike, _normalize_device
from .dtype import normalize_dtype as _as_dtype
from ._ffi.authoring import (
    AttentionAttrs,
    AxisAttrs,
    CastAttrs,
    Conv1DAttrs,
    Conv2DAttrs,
    ElemwiseAttrs,
    MatmulAttrs,
    MhaCachedAttrs,
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
    builder_mark_output,
    builder_name,
    builder_param,
    builder_value_dtype,
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
from .types import Shape

if TYPE_CHECKING:
    from .model import LoadedModel

# A weight source: an existing Tensor, a numpy array, or a nested Python list.
WeightData = object
# Which input axes are dynamic (vary at runtime): a sequence of axis indices
# (auto-named symbols) or a {axis: symbol_name} mapping (reuse a name to tie axes
# across inputs to the same runtime size). The declared int at each axis is the
# authoring placeholder.
DynamicAxes = Union[None, Sequence[int], Mapping[int, str]]
# What `compile`/`export` accept as outputs: one value, a list, or {name: value}.
OutputsLike = Union["Value", Sequence["Value"], Mapping[str, "Value"]]
_PAD_MODES = {
    "zero": AionPadMode.AION_PAD_ZERO,
    "reflect": AionPadMode.AION_PAD_REFLECT,
}

class Value:
    """A builder-bound graph value (an opaque u32 id in one `Builder`).

    This is the **low-level** authoring handle returned by `Builder` ops and by
    `nn` layers. It supports the graph operators/fluent forms and authoring-time
    `shape`/`dtype` introspection (eager per-op inference). The seamless
    high-level lazy value users reach for is `aion.Tensor` (see `tensor.py`),
    which lowers to this at compile time. `Builder` is the explicit escape hatch.
    """

    __slots__ = ("_b", "id", "_name")

    def __init__(self, builder: "Builder", vid: ValueId | int):
        self._b = builder
        self.id: ValueId = ValueId(int(vid))
        self._name = None

    # --- operators ---------------------------------------------------------
    def __matmul__(self, other: "Value") -> "Value":
        return self._b.matmul(self, other)

    def __add__(self, other: "Value") -> "Value":
        return self._b.add(self, other)

    def __sub__(self, other: "Value") -> "Value":
        return self._b.sub(self, other)

    def __mul__(self, other: "Value") -> "Value":
        return self._b.mul(self, other)

    def __truediv__(self, other: "Value") -> "Value":
        return self._b.div(self, other)

    # --- fluent method forms ----------------------------------------------
    def relu(self) -> "Value":
        return self._b.unary("relu", self)

    def gelu(self) -> "Value":
        return self._b.unary("gelu", self)

    def silu(self) -> "Value":
        return self._b.unary("silu", self)

    def sigmoid(self) -> "Value":
        return self._b.unary("sigmoid", self)

    def tanh(self) -> "Value":
        return self._b.unary("tanh", self)

    def softmax(self, axis: int = -1) -> "Value":
        return self._b.softmax(self, axis)

    def reshape(self, shape: Sequence[int]) -> "Value":
        return self._b.reshape(self, shape)

    def transpose2d(self) -> "Value":
        return self._b.transpose2d(self)

    def cast(self, dtype: object) -> "Value":
        return self._b.cast(self, dtype)

    def rename(self, name: str) -> "Value":
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
    def dtype(self) -> AionDType:
        return builder_value_dtype(
            self._b._ctx_owner.ptr, self._b.ptr, self.id
        )

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __repr__(self) -> str:
        return f"Value(id={self.id})"


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
        # dim-symbol name -> authoring placeholder size (the declared axis size),
        # used to resolve symbolic slice/reshape dims to a concrete placeholder.
        self._symbol_placeholders: dict[str, int] = {}
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
        dtype: Union[str, AionDType] = "f32",
        dynamic: DynamicAxes = None,
    ) -> Value:
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
        v = Value(self, value_id)
        for axis, name in symbolic:
            self._add_dim_symbol(v, axis, name)
            # Record the declared placeholder so symbolic view dims can resolve it.
            self._symbol_placeholders.setdefault(name, dims[axis])
        return v

    def _auto_symbol_name(self) -> str:
        name = f"sym{self._symbol_counter}"
        self._symbol_counter += 1
        return name

    def _add_dim_symbol(self, value: Value, axis: int, name: str) -> None:
        builder_add_dim_symbol(
            self._ctx_owner.ptr, self.ptr, value.id, axis, name
        )

    def param(
        self,
        data: WeightData,
        *,
        dtype: Union[str, AionDType] = "f32",
        shape: Optional[Shape] = None,
        quant_axis: Optional[int] = None,
    ) -> Value:
        """Bind a weight from data (a `Tensor`, numpy array, or nested list).

        A float `dtype` binds the data directly; a quantized `dtype`
        (``"q8_0"``) quantizes it in the core first, blocking along `quant_axis`
        (default: the matmul-B reduction axis, rank-2; pass the last axis for an
        embedding table). When `data` is already a `Tensor`, `dtype`/`shape`/
        `quant_axis` are ignored.
        """
        if isinstance(data, Tensor):
            t = data
        else:
            dt = _as_dtype(dtype)
            if dt in (AionDType.AION_DTYPE_Q8_0, AionDType.AION_DTYPE_Q4_0):
                if shape is None:
                    shape = _infer_shape(data)
                t = Tensor.quantize(self._ctx_owner, shape, data, dtype=dt, quant_axis=quant_axis)
            else:
                t = Tensor(data, ctx=self._ctx_owner, dtype=dt)
        self._params.append(t)

        value_id = builder_param(self._ctx_owner.ptr, self.ptr, t.ptr)
        return Value(self, value_id)

    def name(self, value: Value, name: str) -> Value:
        builder_name(self._ctx_owner.ptr, self.ptr, value.id, name)
        return value

    # --- op emit -----------------------------------------------------------
    def _emit(
        self,
        op: AionOp,
        inputs: Sequence[Value],
        attrs: OpAttrs = None,
    ) -> Value:
        value_id = emit_op(
            self._ctx_owner.ptr,
            self.ptr,
            op,
            [value.id for value in inputs],
            attrs,
        )
        return Value(self, value_id)

    # Structured ops.
    def matmul(self, a: Value, b: Value, alpha: float = 1.0, beta: float = 0.0) -> Value:
        return self._emit(
            AionOp.AION_OP_MATMUL, (a, b), MatmulAttrs(alpha, beta)
        )

    def matmul_nt(self, a: Value, b: Value, alpha: float = 1.0, beta: float = 0.0) -> Value:
        return self._emit(
            AionOp.AION_OP_MATMUL_NT, (a, b), MatmulAttrs(alpha, beta)
        )

    def _elemwise(self, op: AionBinaryOp, a: Value, b: Value) -> Value:
        return self._emit(
            AionOp.AION_OP_ELEMWISE, (a, b), ElemwiseAttrs(int(op))
        )

    def add(self, a: Value, b: Value) -> Value:
        return self._elemwise(AionBinaryOp.AION_BINARY_ADD, a, b)

    def sub(self, a: Value, b: Value) -> Value:
        return self._elemwise(AionBinaryOp.AION_BINARY_SUB, a, b)

    def mul(self, a: Value, b: Value) -> Value:
        return self._elemwise(AionBinaryOp.AION_BINARY_MUL, a, b)

    def div(self, a: Value, b: Value) -> Value:
        return self._elemwise(AionBinaryOp.AION_BINARY_DIV, a, b)

    def broadcast_add(self, a: Value, b: Value) -> Value:
        return self._emit(
            AionOp.AION_OP_BROADCAST_LAST_DIM,
            (a, b),
            ElemwiseAttrs(int(AionBinaryOp.AION_BINARY_ADD)),
        )

    def gelu_mul(self, a: Value, b: Value) -> Value:
        return self._emit(AionOp.AION_OP_GELU_MUL, (a, b))

    _UNARY = {
        "relu": AionUnaryOp.AION_UNARY_RELU,
        "gelu": AionUnaryOp.AION_UNARY_GELU,
        "silu": AionUnaryOp.AION_UNARY_SILU,
        "sigmoid": AionUnaryOp.AION_UNARY_SIGMOID,
        "tanh": AionUnaryOp.AION_UNARY_TANH,
        "sqrt": AionUnaryOp.AION_UNARY_SQRT,
        "log": AionUnaryOp.AION_UNARY_LOG,
    }

    def unary(self, op: str, a: Value) -> Value:
        u = self._UNARY[op]
        return self._emit(AionOp.AION_OP_UNARY, (a,), UnaryAttrs(int(u)))

    def softmax(self, a: Value, axis: int = -1) -> Value:
        return self._emit(AionOp.AION_OP_SOFTMAX, (a,), AxisAttrs(axis))

    def rmsnorm(self, x: Value, gamma: Value, beta: Value, *, eps: float = 1e-6, normalized_shape: Sequence[int]) -> Value:
        return self._norm(AionOp.AION_OP_RMSNORM, x, gamma, beta, eps, normalized_shape)

    def layernorm(self, x: Value, gamma: Value, beta: Value, *, eps: float = 1e-5, normalized_shape: Sequence[int]) -> Value:
        return self._norm(AionOp.AION_OP_LAYERNORM, x, gamma, beta, eps, normalized_shape)

    def _norm(
        self,
        op: AionOp,
        x: Value,
        gamma: Value,
        beta: Value,
        eps: float,
        normalized_shape: Sequence[int],
    ) -> Value:
        attrs = NormAttrs(
            float(eps), tuple(int(dim) for dim in normalized_shape)
        )
        return self._emit(op, (x, gamma, beta), attrs)

    def gather_rows(self, table: Value, indices: Value) -> Value:
        return self._emit(AionOp.AION_OP_GATHER_ROWS, (table, indices))

    def rope1d(self, x: Value, positions: Value, *, base_frequency: float, scale_factor: float = 1.0, rope_proportion: float = 1.0) -> Value:
        attrs = Rope1DAttrs(base_frequency, scale_factor, rope_proportion)
        return self._emit(AionOp.AION_OP_ROPE1D, (x, positions), attrs)

    def concat(self, values: Sequence[Value], axis: int) -> Value:
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
                if d not in self._symbol_placeholders:
                    raise ValueError(
                        f"unknown dim symbol {d!r}; declare it on an input via "
                        f"input(..., dynamic={{axis: {d!r}}})")
                concrete.append(int(self._symbol_placeholders[d]))
                names.append(d)
            else:
                concrete.append(int(d))
                names.append(None)
        return ViewDims(tuple(concrete), tuple(names))

    def reshape(self, a: Value, shape: Sequence[Union[int, str]]) -> Value:
        return self._emit(
            AionOp.AION_OP_RESHAPE,
            (a,),
            ReshapeAttrs(self._resolve_view_dims(shape)),
        )

    def transpose2d(self, a: Value) -> Value:
        return self._emit(AionOp.AION_OP_TRANSPOSE2D, (a,))

    def cast(self, a: Value, dtype: object) -> Value:
        dt = _as_dtype(dtype)
        return self._emit(AionOp.AION_OP_CAST, (a,), CastAttrs(dt))

    def argmax(self, a: Value, axis: int = -1) -> Value:
        return self._emit(AionOp.AION_OP_ARGMAX, (a,), AxisAttrs(axis))

    def reduce(self, op: str, a: Value, axis: Optional[int] = None) -> Value:
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

    def compare(self, op: str, a: Value, b: Value) -> Value:
        """Elementwise comparison (`eq/ne/lt/gt/le/ge`) producing i32 {0,1}."""
        return self._elemwise(self._COMPARE[op], a, b)

    # --- broadcast (last-dim vector) binary variants ----------------------
    def broadcast_mul(self, a: Value, vec: Value) -> Value:
        return self._emit(
            AionOp.AION_OP_BROADCAST_LAST_DIM, (a, vec),
            ElemwiseAttrs(int(AionBinaryOp.AION_BINARY_MUL)))

    def broadcast_sub(self, a: Value, vec: Value) -> Value:
        return self._emit(
            AionOp.AION_OP_BROADCAST_LAST_DIM, (a, vec),
            ElemwiseAttrs(int(AionBinaryOp.AION_BINARY_SUB)))

    def broadcast_div(self, a: Value, vec: Value) -> Value:
        return self._emit(
            AionOp.AION_OP_BROADCAST_LAST_DIM, (a, vec),
            ElemwiseAttrs(int(AionBinaryOp.AION_BINARY_DIV)))

    # --- convolutions -----------------------------------------------------
    def conv1d(self, x: Value, weight: Value, bias: Optional[Value] = None, *,
               stride: int = 1, dilation: int = 1, pad_left: int = 0, pad_right: int = 0,
               groups: int = 1, pad_mode: str = "zero") -> Value:
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

    def conv2d(self, x: Value, weight: Value, bias: Optional[Value] = None, *,
               stride_h: int = 1, stride_w: int = 1, dilation_h: int = 1, dilation_w: int = 1,
               pad_top: int = 0, pad_bottom: int = 0, pad_left: int = 0, pad_right: int = 0,
               groups: int = 1, pad_mode: str = "zero") -> Value:
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
    def stft(self, signal: Value, window: Value, *, n_fft: int, hop_length: int,
             center: bool = False) -> Value:
        return self._emit(
            AionOp.AION_OP_STFT,
            (signal, window),
            StftAttrs(n_fft, hop_length, center),
        )

    def rfft(self, x: Value) -> Value:
        return self._emit(AionOp.AION_OP_RFFT, (x,))

    # --- views ------------------------------------------------------------
    def squeeze(self, a: Value, axis: Optional[int] = None) -> Value:
        return self._emit(
            AionOp.AION_OP_SQUEEZE, (a,), OptionalAxisAttrs(axis)
        )

    def unsqueeze(self, a: Value, axis: int) -> Value:
        return self._emit(
            AionOp.AION_OP_UNSQUEEZE, (a,), AxisAttrs(axis)
        )

    def slice(self, a: Value, starts: Sequence[int], lens: Sequence[Union[int, str]]) -> Value:
        """N-D slice (one `start`/`len` per axis). A `len` may be an int (constant)
        or a dim-symbol name (declared via `input(dynamic=...)`) for a symbolic axis."""
        starts_l = [int(s) for s in starts]
        if len(starts_l) != len(lens):
            raise ValueError("slice starts and lens must have equal length")
        attrs = SliceAttrs(
            tuple(starts_l), self._resolve_view_dims(lens)
        )
        return self._emit(AionOp.AION_OP_SLICE, (a,), attrs)

    # --- attention --------------------------------------------------------
    def attention(self, q: Value, k: Value, v: Value, *, scale: float, causal: bool = False) -> Value:
        return self._emit(
            AionOp.AION_OP_ATTENTION,
            (q, k, v),
            AttentionAttrs(scale, causal),
        )

    def mha(self, q: Value, k: Value, v: Value, *, scale: float, causal: bool, heads: int) -> Value:
        return self._emit(
            AionOp.AION_OP_MHA,
            (q, k, v),
            AttentionAttrs(scale, causal, heads),
        )

    def mha_cached(self, q: Value, k: Value, v: Value, positions: Value, end_index: Value, *,
                   scale: float, causal: bool = True, sliding_window: int = 0,
                   attn_logits_soft_cap: float = 0.0) -> Value:
        attrs = MhaCachedAttrs(
            scale, causal, sliding_window, attn_logits_soft_cap
        )
        return self._emit(
            AionOp.AION_OP_MHA_CACHED,
            (q, k, v, positions, end_index),
            attrs,
        )

    def relpos_mha(self, q: Value, k: Value, v: Value, pos_emb: Value, bu: Value, bv: Value,
                   mask: Optional[Value] = None, *, scale: float, heads: int) -> Value:
        inputs: list[Value] = [q, k, v, pos_emb, bu, bv]
        if mask is not None:
            inputs.append(mask)

        return self._emit(
            AionOp.AION_OP_RELPOS_MHA,
            inputs,
            RelposMhaAttrs(scale, heads),
        )

    # --- recurrent / indexed / misc ---------------------------------------
    def lstm_cell(self, x: Value, h: Value, c: Value, w_ih: Value, w_hh: Value,
                  b_ih: Optional[Value] = None, b_hh: Optional[Value] = None) -> Value:
        """One LSTM step -> `[batch, 2*hidden]` = (h_t | c_t). Biases are both-or-neither."""
        if (b_ih is None) != (b_hh is None):
            raise ValueError("lstm_cell requires both b_ih and b_hh or neither")
        inputs: list[Value] = [x, h, c, w_ih, w_hh]
        if b_ih is not None and b_hh is not None:
            inputs.extend((b_ih, b_hh))
        return self._emit(AionOp.AION_OP_LSTM_CELL, inputs)

    def sequence_append(self, cache: Value, new_kv: Value, end_index: Value) -> Value:
        return self._emit(AionOp.AION_OP_SEQUENCE_APPEND, (cache, new_kv, end_index))

    def scatter_row(self, buf: Value, index: Value, src: Value) -> Value:
        return self._emit(AionOp.AION_OP_SCATTER_ROW, (buf, index, src))

    def copy(self, a: Value) -> Value:
        return self._emit(AionOp.AION_OP_COPY, (a,))

    # --- output aliases & input roles -------------------------------------
    def add_output_alias(self, input_value: Value, output_value: Value) -> None:
        """Alias a graph input to an output (io-aliased recurrent state write-back).

        The output must already be marked (see `mark_output`)."""
        builder_add_output_alias(
            self._ctx_owner.ptr,
            self.ptr,
            input_value.id,
            output_value.id,
        )

    def add_input_role(self, value: Value, kind: AionInputRoleKind, *,
                       axis: Optional[int] = None, capacity_symbol: Optional[str] = None,
                       zero_init: bool = True, growable: bool = False, ring: bool = False) -> None:
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
            ring=ring,
        )

    # --- control flow (regions) -------------------------------------------
    def begin_region(self) -> None:
        """Start a region (an `if` branch body or `loop` body). No nesting."""
        builder_begin_region(self._ctx_owner.ptr, self.ptr)

    def end_region(self, outputs: Sequence[Value]) -> RegionId:
        """Close the active region; returns a region id for `if_`/`loop`."""
        return builder_end_region(
            self._ctx_owner.ptr,
            self.ptr,
            [value.id for value in outputs],
        )

    def if_(
        self, cond: Value, then_region: RegionId, else_region: RegionId
    ) -> Value:
        """Single-output conditional over an i32 `[1]` predicate `cond`."""
        value_id = builder_if(
            self._ctx_owner.ptr,
            self.ptr,
            cond.id,
            then_region,
            else_region,
        )
        return Value(self, value_id)

    @overload
    def loop(self, carried: Value, body_region: RegionId, trip: int, *,
             cond_carry: Optional[int] = None, check_before: bool = True) -> Value: ...
    @overload
    def loop(self, carried: Sequence[Value], body_region: RegionId, trip: int, *,
             cond_carry: Optional[int] = None, check_before: bool = True) -> list[Value]: ...

    def loop(
        self,
        carried: Union[Value, Sequence[Value]],
        body_region: RegionId,
        trip: int,
        *,
        cond_carry: Optional[int] = None,
        check_before: bool = True,
    ) -> Union[Value, list[Value]]:
        """Run `body_region` up to `trip` times, threading the carried value(s).

        Pass a single `Value` (returns a `Value`) or a sequence for a multi-carry
        loop (returns a `list`). `cond_carry` names the carry index holding the i32
        `[1]` continue predicate, if any.
        """
        single = isinstance(carried, Value)
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
        values = [Value(self, value_id) for value_id in output_ids]
        return values[0] if single else values

    # --- declarations & terminals -----------------------------------------
    def mark_output(self, value: Value, name: str) -> None:
        builder_mark_output(
            self._ctx_owner.ptr, self.ptr, value.id, name
        )

    def add_metadata(self, key: str, value: str) -> None:
        builder_add_metadata(self._ctx_owner.ptr, self.ptr, key, value)

    def _mark_outputs(self, outputs: OutputsLike) -> None:
        # A `{name: value}` mapping names outputs explicitly; otherwise a value's
        # own `.rename(...)` (its `_name`) is the default, falling back to the
        # positional `output{i}`.
        if isinstance(outputs, Value):
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
