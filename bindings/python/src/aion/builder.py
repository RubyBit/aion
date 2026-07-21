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

from typing import TYPE_CHECKING, Any, Callable, List, Mapping, Optional, Sequence, Union, overload

from .device import DeviceLike, _normalize_device
from .dtype import normalize_dtype as _as_dtype
from .errors import raise_for_status
from ._ffi import ffi, lib
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
from .types import ArrayLike, Shape

if TYPE_CHECKING:
    from .model import LoadedModel

# A weight source: an existing Tensor, a numpy array, or a nested Python list.
WeightData = Union["Tensor", ArrayLike]
# Which input axes are dynamic (vary at runtime): a sequence of axis indices
# (auto-named symbols) or a {axis: symbol_name} mapping (reuse a name to tie axes
# across inputs to the same runtime size). The declared int at each axis is the
# authoring placeholder.
DynamicAxes = Union[None, Sequence[int], Mapping[int, str]]
# What `compile`/`export` accept as outputs: one value, a list, or {name: value}.
OutputsLike = Union["Value", Sequence["Value"], Mapping[str, "Value"]]
# Per-op attribute configurator invoked with the spec's `attr` union.
AttrConfigure = Callable[[Any], None]

_PAD_MODES = {
    "zero": AionPadMode.AION_PAD_ZERO,
    "reflect": AionPadMode.AION_PAD_REFLECT,
}

class Value:
    """A graph value produced/consumed inside a `Builder` (an opaque u32 id)."""

    __slots__ = ("_b", "id")

    def __init__(self, builder: "Builder", vid: int):
        self._b = builder
        self.id = int(vid)

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

    def cast(self, dtype) -> "Value":
        return self._b.cast(self, dtype)

    def rename(self, name: str) -> "Value":
        self._b.name(self, name)
        return self

    # --- authoring-time introspection -------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        """Shape at the authoring placeholder sizes (eager per-op inference).

        Concrete ints throughout — a dynamic axis reports the placeholder it was
        declared with, propagated to derived values. The compiled/exported model
        still serves any size on dynamic axes.
        """
        rank = ffi.new("size_t*")
        st = lib.aion_builder_value_rank(self._b.ptr, self.id, rank)
        raise_for_status(st, self._b._ctx_owner.ptr, what="aion_builder_value_rank")
        n = int(rank[0])
        dims = ffi.new("size_t[]", n) if n > 0 else ffi.NULL
        st = lib.aion_builder_value_shape(self._b.ptr, self.id, dims, n)
        raise_for_status(st, self._b._ctx_owner.ptr, what="aion_builder_value_shape")
        return tuple(int(dims[i]) for i in range(n))

    @property
    def dtype(self) -> AionDType:
        out = ffi.new("AionDType*")
        st = lib.aion_builder_value_dtype(self._b.ptr, self.id, out)
        raise_for_status(st, self._b._ctx_owner.ptr, what="aion_builder_value_dtype")
        return AionDType(int(out[0]))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __repr__(self) -> str:
        return f"Value(id={self.id})"


class Builder:
    """Authors a graph, then compiles or exports it."""

    def __init__(self, ctx=None):
        from .context import get_default_context

        if ctx is None:
            ctx = get_default_context()
        self._ctx_owner = ctx
        out = ffi.new("AionBuilder**")
        st = lib.aion_builder_create(ctx.ptr, out)
        raise_for_status(st, ctx.ptr, what="aion_builder_create")
        self._b = out[0]
        self._closed = False
        # Keep param tensors alive for the builder's lifetime.
        self._params: list[Tensor] = []
        self._symbol_counter = 0
        # dim-symbol name -> authoring placeholder size (the declared axis size),
        # used to resolve symbolic slice/reshape dims to a concrete placeholder.
        self._symbol_placeholders: dict[str, int] = {}
        ctx._register_child(self)

    @property
    def ptr(self):
        return self._b

    @property
    def context(self):
        return self._ctx_owner

    def close(self) -> None:
        if self._closed:
            return
        lib.aion_builder_destroy(self._b)
        try:
            self._ctx_owner._unregister_child(self)
        except Exception:
            pass
        self._b = ffi.NULL
        self._closed = True
        self._params.clear()

    def __enter__(self) -> "Builder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):  # pragma: no cover
        try:
            if not getattr(self, "_closed", True):
                lib.aion_builder_destroy(getattr(self, "_b", ffi.NULL))
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

        c_shape = ffi.new("size_t[]", dims)
        out = ffi.new("AionValueId*")
        st = lib.aion_builder_input(self._b, int(dt), len(dims), c_shape, out)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_builder_input")
        v = Value(self, out[0])
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
        st = lib.aion_builder_add_dim_symbol(self._b, value.id, int(axis), _c_str(name))
        raise_for_status(st, self._ctx_owner.ptr, what="aion_builder_add_dim_symbol")

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

        out = ffi.new("AionValueId*")
        st = lib.aion_builder_param(self._b, t.ptr, out)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_builder_param")
        return Value(self, out[0])

    def name(self, value: Value, name: str) -> Value:
        st = lib.aion_builder_name(self._b, value.id, _c_str(name))
        raise_for_status(st, self._ctx_owner.ptr, what="aion_builder_name")
        return value

    # --- op emit -----------------------------------------------------------
    def _emit(self, op: AionOp, inputs: Sequence[Value], configure: Optional[AttrConfigure] = None) -> Value:
        spec = ffi.new("AionOpSpec*")
        spec.op = int(op)
        ids = ffi.new("AionValueId[]", [int(v.id) for v in inputs])
        spec.inputs = ids
        spec.inputs_len = len(inputs)
        if configure is not None:
            configure(spec.attr)
        out = ffi.new("AionValueId*")
        st = lib.aion_builder_op(self._b, spec, out)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_builder_op")
        return Value(self, out[0])

    # Structured ops.
    def matmul(self, a: Value, b: Value, alpha: float = 1.0, beta: float = 0.0) -> Value:
        def cfg(attr):
            attr.matmul.alpha = float(alpha)
            attr.matmul.beta = float(beta)

        return self._emit(AionOp.AION_OP_MATMUL, (a, b), cfg)

    def matmul_nt(self, a: Value, b: Value, alpha: float = 1.0, beta: float = 0.0) -> Value:
        def cfg(attr):
            attr.matmul.alpha = float(alpha)
            attr.matmul.beta = float(beta)

        return self._emit(AionOp.AION_OP_MATMUL_NT, (a, b), cfg)

    def _elemwise(self, op: AionBinaryOp, a: Value, b: Value) -> Value:
        return self._emit(AionOp.AION_OP_ELEMWISE, (a, b), lambda attr: setattr(attr.elemwise, "op", int(op)))

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
            lambda attr: setattr(attr.elemwise, "op", int(AionBinaryOp.AION_BINARY_ADD)),
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
        return self._emit(AionOp.AION_OP_UNARY, (a,), lambda attr: setattr(attr.unary, "op", int(u)))

    def softmax(self, a: Value, axis: int = -1) -> Value:
        return self._emit(AionOp.AION_OP_SOFTMAX, (a,), lambda attr: setattr(attr.softmax, "axis", int(axis)))

    def rmsnorm(self, x: Value, gamma: Value, beta: Value, *, eps: float = 1e-6, normalized_shape: Sequence[int]) -> Value:
        return self._norm(AionOp.AION_OP_RMSNORM, x, gamma, beta, eps, normalized_shape)

    def layernorm(self, x: Value, gamma: Value, beta: Value, *, eps: float = 1e-5, normalized_shape: Sequence[int]) -> Value:
        return self._norm(AionOp.AION_OP_LAYERNORM, x, gamma, beta, eps, normalized_shape)

    def _norm(self, op, x, gamma, beta, eps, normalized_shape) -> Value:
        ns = ffi.new("size_t[]", [int(d) for d in normalized_shape])

        def cfg(attr):
            attr.norm.eps = float(eps)
            attr.norm.normalized_shape = ns
            attr.norm.normalized_shape_len = len(normalized_shape)

        # `ns` must outlive the call; _emit runs synchronously so the closure
        # (and thus `ns`) is alive throughout.
        return self._emit(op, (x, gamma, beta), cfg)

    def gather_rows(self, table: Value, indices: Value) -> Value:
        return self._emit(AionOp.AION_OP_GATHER_ROWS, (table, indices))

    def rope1d(self, x: Value, positions: Value, *, base_frequency: float, scale_factor: float = 1.0, rope_proportion: float = 1.0) -> Value:
        def cfg(attr):
            attr.rope1d.base_frequency = float(base_frequency)
            attr.rope1d.scale_factor = float(scale_factor)
            attr.rope1d.rope_proportion = float(rope_proportion)

        return self._emit(AionOp.AION_OP_ROPE1D, (x, positions), cfg)

    def concat(self, values: Sequence[Value], axis: int) -> Value:
        return self._emit(AionOp.AION_OP_CONCAT, tuple(values), lambda attr: setattr(attr.concat, "axis", int(axis)))

    def _resolve_view_dims(self, dims: Sequence[Union[int, str]]):
        """Resolve a view-op dim list (ints and dim-symbol names) to a concrete
        placeholder `size_t[]` plus an optional `char*[]` of per-axis symbol names
        (NULL where constant). Returns (concrete_cdata, symbols_cdata_or_NULL,
        keepalive) — the keepalive holds the individual C strings alive."""
        concrete: list[int] = []
        names: list[Optional[str]] = []
        any_sym = False
        for d in dims:
            if isinstance(d, str):
                if d not in self._symbol_placeholders:
                    raise ValueError(
                        f"unknown dim symbol {d!r}; declare it on an input via "
                        f"input(..., dynamic={{axis: {d!r}}})")
                concrete.append(int(self._symbol_placeholders[d]))
                names.append(d)
                any_sym = True
            else:
                concrete.append(int(d))
                names.append(None)
        c_concrete = ffi.new("size_t[]", concrete)
        if not any_sym:
            return c_concrete, ffi.NULL, [c_concrete]
        keep: list[Any] = [c_concrete]
        ptrs = []
        for nm in names:
            if nm is None:
                ptrs.append(ffi.NULL)
            else:
                cs = _c_str(nm)
                keep.append(cs)
                ptrs.append(cs)
        c_syms = ffi.new("char*[]", ptrs)
        keep.append(c_syms)
        return c_concrete, c_syms, keep

    def reshape(self, a: Value, shape: Sequence[Union[int, str]]) -> Value:
        c_shape, c_syms, _keep = self._resolve_view_dims(shape)

        def cfg(attr):
            attr.reshape.shape = c_shape
            attr.reshape.shape_len = len(shape)
            attr.reshape.shape_symbols = c_syms

        # `_keep` holds the cdata alive across the synchronous _emit call.
        return self._emit(AionOp.AION_OP_RESHAPE, (a,), cfg)

    def transpose2d(self, a: Value) -> Value:
        return self._emit(AionOp.AION_OP_TRANSPOSE2D, (a,))

    def cast(self, a: Value, dtype) -> Value:
        dt = _as_dtype(dtype)
        return self._emit(AionOp.AION_OP_CAST, (a,), lambda attr: setattr(attr.cast, "to_dtype", int(dt)))

    def argmax(self, a: Value, axis: int = -1) -> Value:
        return self._emit(AionOp.AION_OP_ARGMAX, (a,), lambda attr: setattr(attr.argmax, "axis", int(axis)))

    def reduce(self, op: str, a: Value, axis: Optional[int] = None) -> Value:
        r = AionReduceOp.AION_REDUCE_SUM if op == "sum" else AionReduceOp.AION_REDUCE_MEAN

        def cfg(attr):
            attr.reduce.op = int(r)
            attr.reduce.axis = int(axis) if axis is not None else 0
            attr.reduce.has_axis = 1 if axis is not None else 0

        return self._emit(AionOp.AION_OP_REDUCE, (a,), cfg)

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
            lambda attr: setattr(attr.elemwise, "op", int(AionBinaryOp.AION_BINARY_MUL)))

    def broadcast_sub(self, a: Value, vec: Value) -> Value:
        return self._emit(
            AionOp.AION_OP_BROADCAST_LAST_DIM, (a, vec),
            lambda attr: setattr(attr.elemwise, "op", int(AionBinaryOp.AION_BINARY_SUB)))

    def broadcast_div(self, a: Value, vec: Value) -> Value:
        return self._emit(
            AionOp.AION_OP_BROADCAST_LAST_DIM, (a, vec),
            lambda attr: setattr(attr.elemwise, "op", int(AionBinaryOp.AION_BINARY_DIV)))

    # --- convolutions -----------------------------------------------------
    def conv1d(self, x: Value, weight: Value, bias: Optional[Value] = None, *,
               stride: int = 1, dilation: int = 1, pad_left: int = 0, pad_right: int = 0,
               groups: int = 1, pad_mode: str = "zero") -> Value:
        inputs = (x, weight) if bias is None else (x, weight, bias)

        def cfg(attr):
            attr.conv1d.stride = int(stride)
            attr.conv1d.dilation = int(dilation)
            attr.conv1d.pad_left = int(pad_left)
            attr.conv1d.pad_right = int(pad_right)
            attr.conv1d.groups = int(groups)
            attr.conv1d.pad_mode = int(_PAD_MODES[pad_mode])

        return self._emit(AionOp.AION_OP_CONV1D, inputs, cfg)

    def conv2d(self, x: Value, weight: Value, bias: Optional[Value] = None, *,
               stride_h: int = 1, stride_w: int = 1, dilation_h: int = 1, dilation_w: int = 1,
               pad_top: int = 0, pad_bottom: int = 0, pad_left: int = 0, pad_right: int = 0,
               groups: int = 1, pad_mode: str = "zero") -> Value:
        inputs = (x, weight) if bias is None else (x, weight, bias)

        def cfg(attr):
            attr.conv2d.stride_h = int(stride_h)
            attr.conv2d.stride_w = int(stride_w)
            attr.conv2d.dilation_h = int(dilation_h)
            attr.conv2d.dilation_w = int(dilation_w)
            attr.conv2d.pad_top = int(pad_top)
            attr.conv2d.pad_bottom = int(pad_bottom)
            attr.conv2d.pad_left = int(pad_left)
            attr.conv2d.pad_right = int(pad_right)
            attr.conv2d.groups = int(groups)
            attr.conv2d.pad_mode = int(_PAD_MODES[pad_mode])

        return self._emit(AionOp.AION_OP_CONV2D, inputs, cfg)

    # --- signal (FFT) -----------------------------------------------------
    def stft(self, signal: Value, window: Value, *, n_fft: int, hop_length: int,
             center: bool = False) -> Value:
        def cfg(attr):
            attr.stft.n_fft = int(n_fft)
            attr.stft.hop_length = int(hop_length)
            attr.stft.center = 1 if center else 0

        return self._emit(AionOp.AION_OP_STFT, (signal, window), cfg)

    def rfft(self, x: Value) -> Value:
        return self._emit(AionOp.AION_OP_RFFT, (x,))

    # --- views ------------------------------------------------------------
    def squeeze(self, a: Value, axis: Optional[int] = None) -> Value:
        def cfg(attr):
            attr.squeeze.axis = int(axis) if axis is not None else 0
            attr.squeeze.has_axis = 1 if axis is not None else 0

        return self._emit(AionOp.AION_OP_SQUEEZE, (a,), cfg)

    def unsqueeze(self, a: Value, axis: int) -> Value:
        return self._emit(AionOp.AION_OP_UNSQUEEZE, (a,),
                          lambda attr: setattr(attr.unsqueeze, "axis", int(axis)))

    def slice(self, a: Value, starts: Sequence[int], lens: Sequence[Union[int, str]]) -> Value:
        """N-D slice (one `start`/`len` per axis). A `len` may be an int (constant)
        or a dim-symbol name (declared via `input(dynamic=...)`) for a symbolic axis."""
        starts_l = [int(s) for s in starts]
        if len(starts_l) != len(lens):
            raise ValueError("slice starts and lens must have equal length")
        c_starts = ffi.new("size_t[]", starts_l)
        c_lens, c_syms, _keep = self._resolve_view_dims(lens)

        def cfg(attr):
            attr.slice.starts = c_starts
            attr.slice.lens = c_lens
            attr.slice.len = len(starts_l)
            attr.slice.len_symbols = c_syms

        # c_starts/c_lens/_keep must outlive the call; kept via locals + closure.
        return self._emit(AionOp.AION_OP_SLICE, (a,), cfg)

    # --- attention --------------------------------------------------------
    def attention(self, q: Value, k: Value, v: Value, *, scale: float, causal: bool = False) -> Value:
        def cfg(attr):
            attr.attention.scale = float(scale)
            attr.attention.causal = 1 if causal else 0

        return self._emit(AionOp.AION_OP_ATTENTION, (q, k, v), cfg)

    def mha(self, q: Value, k: Value, v: Value, *, scale: float, causal: bool, heads: int) -> Value:
        def cfg(attr):
            attr.attention.scale = float(scale)
            attr.attention.causal = 1 if causal else 0
            attr.attention.heads = int(heads)

        return self._emit(AionOp.AION_OP_MHA, (q, k, v), cfg)

    def mha_cached(self, q: Value, k: Value, v: Value, positions: Value, end_index: Value, *,
                   scale: float, causal: bool = True, sliding_window: int = 0,
                   attn_logits_soft_cap: float = 0.0) -> Value:
        def cfg(attr):
            attr.mha_cached.scale = float(scale)
            attr.mha_cached.causal = 1 if causal else 0
            attr.mha_cached.sliding_window = int(sliding_window)
            attr.mha_cached.attn_logits_soft_cap = float(attn_logits_soft_cap)

        return self._emit(AionOp.AION_OP_MHA_CACHED, (q, k, v, positions, end_index), cfg)

    def relpos_mha(self, q: Value, k: Value, v: Value, pos_emb: Value, bu: Value, bv: Value,
                   mask: Optional[Value] = None, *, scale: float, heads: int) -> Value:
        inputs: List[Value] = [q, k, v, pos_emb, bu, bv]
        if mask is not None:
            inputs.append(mask)

        def cfg(attr):
            attr.relpos_mha.scale = float(scale)
            attr.relpos_mha.heads = int(heads)

        return self._emit(AionOp.AION_OP_RELPOS_MHA, inputs, cfg)

    # --- recurrent / indexed / misc ---------------------------------------
    def lstm_cell(self, x: Value, h: Value, c: Value, w_ih: Value, w_hh: Value,
                  b_ih: Optional[Value] = None, b_hh: Optional[Value] = None) -> Value:
        """One LSTM step -> `[batch, 2*hidden]` = (h_t | c_t). Biases are both-or-neither."""
        if (b_ih is None) != (b_hh is None):
            raise ValueError("lstm_cell requires both b_ih and b_hh or neither")
        inputs: List[Value] = [x, h, c, w_ih, w_hh]
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
        st = lib.aion_builder_add_output_alias(self._b, input_value.id, output_value.id)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_builder_add_output_alias")

    def add_input_role(self, value: Value, kind: AionInputRoleKind, *,
                       axis: Optional[int] = None, capacity_symbol: Optional[str] = None,
                       zero_init: bool = True, growable: bool = False, ring: bool = False) -> None:
        """Tag an input with a runtime role (KV cache, positions, tokens, state, …).

        `capacity_symbol` names a dim symbol declared on `value` (via
        `input(dynamic={axis: name})`) whose size the runtime supplies at load."""
        cap = _c_str(capacity_symbol) if capacity_symbol is not None else ffi.NULL
        st = lib.aion_builder_add_input_role(
            self._b, value.id, int(kind),
            int(axis) if axis is not None else 0, 1 if axis is not None else 0,
            cap, 1 if zero_init else 0, 1 if growable else 0, 1 if ring else 0)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_builder_add_input_role")

    # --- control flow (regions) -------------------------------------------
    def begin_region(self) -> None:
        """Start a region (an `if` branch body or `loop` body). No nesting."""
        st = lib.aion_builder_begin_region(self._b)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_builder_begin_region")

    def end_region(self, outputs: Sequence[Value]) -> int:
        """Close the active region; returns a region id for `if_`/`loop`."""
        outs = list(outputs)
        ids = ffi.new("AionValueId[]", [int(v.id) for v in outs])
        out_region = ffi.new("AionRegionId*")
        st = lib.aion_builder_end_region(self._b, ids, len(outs), out_region)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_builder_end_region")
        return int(out_region[0])

    def if_(self, cond: Value, then_region: int, else_region: int) -> Value:
        """Single-output conditional over an i32 `[1]` predicate `cond`."""
        out = ffi.new("AionValueId*")
        st = lib.aion_builder_if(self._b, cond.id, int(then_region), int(else_region), out)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_builder_if")
        return Value(self, out[0])

    @overload
    def loop(self, carried: Value, body_region: int, trip: int, *,
             cond_carry: Optional[int] = None, check_before: bool = True) -> Value: ...
    @overload
    def loop(self, carried: Sequence[Value], body_region: int, trip: int, *,
             cond_carry: Optional[int] = None, check_before: bool = True) -> List[Value]: ...

    def loop(
        self,
        carried: Union[Value, Sequence[Value]],
        body_region: int,
        trip: int,
        *,
        cond_carry: Optional[int] = None,
        check_before: bool = True,
    ) -> Union[Value, List[Value]]:
        """Run `body_region` up to `trip` times, threading the carried value(s).

        Pass a single `Value` (returns a `Value`) or a sequence for a multi-carry
        loop (returns a `list`). `cond_carry` names the carry index holding the i32
        `[1]` continue predicate, if any.
        """
        single = isinstance(carried, Value)
        inits = [carried] if single else list(carried)
        n = len(inits)
        ids = ffi.new("AionValueId[]", [int(v.id) for v in inits])
        outs = ffi.new("AionValueId[]", n)
        st = lib.aion_builder_loop(
            self._b,
            ids,
            n,
            int(body_region),
            int(trip),
            int(cond_carry) if cond_carry is not None else 0,
            1 if cond_carry is not None else 0,
            1 if check_before else 0,
            outs,
        )
        raise_for_status(st, self._ctx_owner.ptr, what="aion_builder_loop")
        values = [Value(self, outs[i]) for i in range(n)]
        return values[0] if single else values

    # --- declarations & terminals -----------------------------------------
    def mark_output(self, value: Value, name: str) -> None:
        st = lib.aion_builder_mark_output(self._b, value.id, _c_str(name))
        raise_for_status(st, self._ctx_owner.ptr, what="aion_builder_mark_output")

    def add_metadata(self, key: str, value: str) -> None:
        st = lib.aion_builder_add_metadata(self._b, _c_str(key), _c_str(value))
        raise_for_status(st, self._ctx_owner.ptr, what="aion_builder_add_metadata")

    def _mark_outputs(self, outputs: OutputsLike) -> None:
        if isinstance(outputs, Value):
            self.mark_output(outputs, "output0")
        elif isinstance(outputs, Mapping):
            for name, v in outputs.items():
                self.mark_output(v, str(name))
        else:
            for i, v in enumerate(outputs):
                self.mark_output(v, f"output{i}")

    def compile(self, outputs: Optional[OutputsLike] = None, *, device: DeviceLike = None) -> "LoadedModel":
        """Compile to an in-process `LoadedModel` (concrete shapes only)."""
        from .model import LoadedModel

        if outputs is not None:
            self._mark_outputs(outputs)
        kind, index = _normalize_device(device)
        out_m = ffi.new("AionLoadedModel**")
        st = lib.aion_builder_compile(self._b, int(kind), int(index), out_m)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_builder_compile")
        return LoadedModel(self._ctx_owner, out_m[0])

    def export(self, path: str, outputs: Optional[OutputsLike] = None) -> None:
        """Serialize the graph to a `.aion` file at `path`."""
        if outputs is not None:
            self._mark_outputs(outputs)
        st = lib.aion_builder_export_path(self._b, _c_str(str(path)))
        raise_for_status(st, self._ctx_owner.ptr, what="aion_builder_export_path")


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


def _c_str(s: str) -> Any:
    return ffi.new("char[]", s.encode("utf-8"))


def _infer_shape(data: WeightData) -> tuple[int, ...]:
    try:
        import numpy as np  # type: ignore
    except ImportError:
        np = None  # type: ignore
    if np is not None and isinstance(data, np.ndarray):
        return tuple(int(x) for x in data.shape)
    from .tensor import _flatten_nested

    shape, _ = _flatten_nested(data)
    return shape
