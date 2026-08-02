# SPDX-License-Identifier: Apache-2.0
"""Layer catalog, mirroring `src/aion/api/nn`.

Weight layouts are Aion's, not any framework's: dense weights are matmul-B
ordered (`[in, out]`), conv1d is `[k, c_in/groups, c_out]` (NLC), conv2d is
`[k_h, k_w, c_in/groups, c_out]` (NHWC). Converting a foreign checkpoint into
that order is the converter's job.
"""
from __future__ import annotations

from typing import Optional, Sequence, TypedDict

from ..builder import TensorRef, WeightData
from ..dtype import float32
from ..types import DTypeLike
from .module import Module, Parameter, builder_of

# `matmul` aligns operand ranks itself, so a plain `[in, out]` weight works
# against a `[batch, seq, in]` activation with no reshape here.


class Conv1DOpts(TypedDict):
    """Exactly `Builder.conv1d`'s keyword arguments, so `**opts` stays typed."""

    stride: int
    dilation: int
    pad_left: int
    pad_right: int
    pad_mode: str
    groups: int


class Conv1DPadding(TypedDict):
    """The subset `causal_pad` fixes; splat into a `Conv1D(...)` call."""

    dilation: int
    pad_left: int
    pad_right: int


class Conv2DOpts(TypedDict):
    """Exactly `Builder.conv2d`'s keyword arguments."""

    stride_h: int
    stride_w: int
    dilation_h: int
    dilation_w: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    pad_mode: str
    groups: int


class Linear(Module):
    """`y = x @ w (+ b)`, with `w` in matmul-B layout `[in, out]`.

    Pass ``dtype=aion.q8_0`` to quantize the weight in the core (bias stays float).
    """

    def __init__(
        self,
        weight: WeightData,
        bias: Optional[WeightData] = None,
        *,
        name: Optional[str] = None,
        dtype: DTypeLike = float32,
        alpha: float = 1.0,
        beta: float = 0.0,
        nt: bool = False,
    ) -> None:
        self._layer_name = name
        self.weight = Parameter(weight, dtype=dtype)
        self.bias = Parameter(bias, dtype=float32) if bias is not None else None
        self.alpha = float(alpha)
        self.beta = float(beta)
        # Contract against the weight's *rows* (`[out, in]`) instead of its
        # columns: what a tied embedding head needs, so one `[vocab, dim]` table
        # serves both the lookup and the output projection.
        self.nt = bool(nt)

    def forward(self, x: TensorRef) -> TensorRef:
        b = builder_of(x)
        with self._scoped(b):
            w = self.weight.value(b, "weight")
            y = b.matmul_nt(x, w, self.alpha, self.beta) if self.nt else b.matmul(x, w, self.alpha, self.beta)
            if self.bias is not None:
                y = b.add(y, self.bias.value(b, "bias"))
            return y


class Embedding(Module):
    """Row lookup into a `[vocab, dim]` table.

    A quantized table blocks along the feature axis (`quant_axis=1`), not the
    matmul reduction axis. `weight_value` hands the bound table back so a tied
    output head can reuse it.
    """

    def __init__(
        self,
        table: WeightData,
        *,
        name: Optional[str] = None,
        dtype: DTypeLike = float32,
    ) -> None:
        self._layer_name = name
        from ..dtype import is_quantized, normalize_dtype

        kwargs = {}
        if is_quantized(normalize_dtype(dtype)):
            kwargs = {"shape": _shape_of(table), "quant_axis": 1}
        self.weight = Parameter(table, dtype=dtype, **kwargs)

    def weight_value(self, b) -> TensorRef:
        with self._scoped(b):
            return self.weight.value(b, "weight")

    def forward(self, ids: TensorRef) -> TensorRef:
        b = builder_of(ids)
        with self._scoped(b):
            return b.gather(self.weight.value(b, "weight"), ids, axis=0)


class _Norm(Module):
    """Shared body for LayerNorm/RMSNorm.

    `gamma`/`beta` are optional: an omitted one becomes the shared identity vector
    the Builder synthesizes (`ones`/`zeros`), so "RMSNorm with no bias" stops being
    the caller's problem even though the op takes both operands unconditionally.
    """

    _default_eps: float = 1e-5

    def __init__(
        self,
        gamma: Optional[WeightData] = None,
        beta: Optional[WeightData] = None,
        *,
        name: Optional[str] = None,
        eps: Optional[float] = None,
        normalized_shape: Optional[Sequence[int]] = None,
    ) -> None:
        self._layer_name = name
        if normalized_shape is not None:
            shape = tuple(int(d) for d in normalized_shape)
        elif gamma is not None:
            shape = _shape_of(gamma)
        else:
            raise ValueError(
                "normalized_shape is required when gamma is omitted: "
                "nothing else determines the normalized width"
            )
        self.normalized_shape = shape
        self.eps = float(eps) if eps is not None else self._default_eps
        self.weight = Parameter(gamma, dtype=float32) if gamma is not None else None
        self.bias = Parameter(beta, dtype=float32) if beta is not None else None

    @property
    def _width(self) -> int:
        n = 1
        for d in self.normalized_shape:
            n *= int(d)
        return n

    def forward(self, x: TensorRef) -> TensorRef:
        b = builder_of(x)
        with self._scoped(b):
            g = self.weight.value(b, "weight") if self.weight is not None else b.ones(self._width)
            bt = self.bias.value(b, "bias") if self.bias is not None else b.zeros(self._width)
            return self._apply(b, x, g, bt)

    def _apply(self, b, x: TensorRef, g: TensorRef, bt: TensorRef) -> TensorRef:  # pragma: no cover
        raise NotImplementedError


class LayerNorm(_Norm):
    """`(x - mean) / sqrt(var + eps) * gamma + beta` over the trailing dims."""

    _default_eps = 1e-5

    def _apply(self, b, x: TensorRef, g: TensorRef, bt: TensorRef) -> TensorRef:
        return b.layernorm(x, g, bt, eps=self.eps, normalized_shape=self.normalized_shape)


class RMSNorm(_Norm):
    """`x / sqrt(mean(x^2) + eps) * gamma + beta` over the trailing dims."""

    _default_eps = 1e-6

    def _apply(self, b, x: TensorRef, g: TensorRef, bt: TensorRef) -> TensorRef:
        return b.rmsnorm(x, g, bt, eps=self.eps, normalized_shape=self.normalized_shape)


class Conv1D(Module):
    """NLC (channel-last) conv1d, weight `[k, c_in/groups, c_out]`."""

    def __init__(
        self,
        weight: WeightData,
        bias: Optional[WeightData] = None,
        *,
        name: Optional[str] = None,
        dtype: DTypeLike = float32,
        stride: int = 1,
        dilation: int = 1,
        pad_left: int = 0,
        pad_right: int = 0,
        pad_mode: str = "zero",
        groups: int = 1,
    ) -> None:
        self._layer_name = name
        self.weight = Parameter(weight, dtype=dtype)
        self.bias = Parameter(bias, dtype=float32) if bias is not None else None
        self.opts: Conv1DOpts = {
            "stride": stride, "dilation": dilation, "pad_left": pad_left,
            "pad_right": pad_right, "pad_mode": pad_mode, "groups": groups,
        }

    def forward(self, x: TensorRef) -> TensorRef:
        b = builder_of(x)
        with self._scoped(b):
            bias = self.bias.value(b, "bias") if self.bias is not None else None
            return b.conv1d(x, self.weight.value(b, "weight"), bias, **self.opts)

    @staticmethod
    def causal_pad(kernel: int, dilation: int = 1) -> Conv1DPadding:
        """Left-only padding of `(k - 1) * dilation`.

        No output frame then depends on a future input frame, which is what a
        streaming model needs and a classic source of train/serve skew when off
        by one.
        """
        if kernel <= 0 or dilation <= 0:
            raise ValueError("kernel and dilation must be > 0")
        return {"dilation": dilation, "pad_left": (kernel - 1) * dilation, "pad_right": 0}


class DepthwiseConv1D(Conv1D):
    """Conv1D with one filter per channel, weight `[k, 1, channels]`.

    Groups are derived from the weight, so the "groups must equal the channel
    count" invariant cannot be set wrong.
    """

    def __init__(
        self,
        weight: WeightData,
        bias: Optional[WeightData] = None,
        *,
        name: Optional[str] = None,
        dtype: DTypeLike = float32,
        stride: int = 1,
        dilation: int = 1,
        pad_left: int = 0,
        pad_right: int = 0,
        pad_mode: str = "zero",
    ) -> None:
        shape = _shape_of(weight)
        if len(shape) != 3 or shape[1] != 1:
            raise ValueError(
                f"depthwise conv1d weight must be [k, 1, channels], got {shape}"
            )
        # `groups` is derived, never passed: that is the invariant this type exists
        # to enforce.
        super().__init__(
            weight, bias, name=name, dtype=dtype, stride=stride, dilation=dilation,
            pad_left=pad_left, pad_right=pad_right, pad_mode=pad_mode,
            groups=int(shape[2]),
        )


class Conv2D(Module):
    """NHWC conv2d, weight `[k_h, k_w, c_in/groups, c_out]`."""

    def __init__(
        self,
        weight: WeightData,
        bias: Optional[WeightData] = None,
        *,
        name: Optional[str] = None,
        dtype: DTypeLike = float32,
        stride_h: int = 1,
        stride_w: int = 1,
        dilation_h: int = 1,
        dilation_w: int = 1,
        pad_top: int = 0,
        pad_bottom: int = 0,
        pad_left: int = 0,
        pad_right: int = 0,
        pad_mode: str = "zero",
        groups: int = 1,
    ) -> None:
        self._layer_name = name
        self.weight = Parameter(weight, dtype=dtype)
        self.bias = Parameter(bias, dtype=float32) if bias is not None else None
        self.opts: Conv2DOpts = {
            "stride_h": stride_h, "stride_w": stride_w,
            "dilation_h": dilation_h, "dilation_w": dilation_w,
            "pad_top": pad_top, "pad_bottom": pad_bottom,
            "pad_left": pad_left, "pad_right": pad_right,
            "pad_mode": pad_mode, "groups": groups,
        }

    def forward(self, x: TensorRef) -> TensorRef:
        b = builder_of(x)
        with self._scoped(b):
            bias = self.bias.value(b, "bias") if self.bias is not None else None
            return b.conv2d(x, self.weight.value(b, "weight"), bias, **self.opts)


class LSTMCell(Module["tuple[TensorRef, TensorRef]"]):
    """Single-timestep LSTM cell.

    Weights are Aion-native: `w_ih [input, 4h]`, `w_hh [h, 4h]`, biases `[4h]`.
    The fused op returns a packed `[batch, 2h]` state; `forward` splits it so
    callers get `(h, c)` rather than re-deriving the offsets.
    """

    def __init__(
        self,
        w_ih: WeightData,
        w_hh: WeightData,
        b_ih: Optional[WeightData] = None,
        b_hh: Optional[WeightData] = None,
        *,
        name: Optional[str] = None,
    ) -> None:
        self._layer_name = name
        if (b_ih is None) != (b_hh is None):
            raise ValueError("LSTM biases come as a pair: provide both or neither")

        ih, hh = _shape_of(w_ih), _shape_of(w_hh)
        if len(ih) != 2 or len(hh) != 2:
            raise ValueError(f"LSTM weights must be rank-2, got {ih} and {hh}")
        gate_dim = int(ih[1])
        if gate_dim % 4 != 0:
            raise ValueError(f"LSTM gate dim must be a multiple of 4, got {gate_dim}")
        hidden = gate_dim // 4
        if hh != (hidden, gate_dim):
            raise ValueError(f"w_hh must be [{hidden}, {gate_dim}], got {hh}")

        self.input_size = int(ih[0])
        self.hidden_size = hidden
        self.weight_ih = Parameter(w_ih, dtype=float32)
        self.weight_hh = Parameter(w_hh, dtype=float32)
        self.bias_ih = Parameter(b_ih, dtype=float32) if b_ih is not None else None
        self.bias_hh = Parameter(b_hh, dtype=float32) if b_hh is not None else None

    def forward(self, x: TensorRef, h: TensorRef, c: TensorRef) -> tuple[TensorRef, TensorRef]:
        b = builder_of(x)
        with self._scoped(b):
            packed = b.lstm_cell(
                x, h, c,
                self.weight_ih.value(b, "weight_ih"),
                self.weight_hh.value(b, "weight_hh"),
                self.bias_ih.value(b, "bias_ih") if self.bias_ih is not None else None,
                self.bias_hh.value(b, "bias_hh") if self.bias_hh is not None else None,
            )
            # As authored, so a free batch axis stays free across the split.
            batch = packed.dims[0]
            hh = self.hidden_size
            return (
                b.slice(packed, (0, 0), (batch, hh)),
                b.slice(packed, (0, hh), (batch, hh)),
            )


class Activation(Module):
    """A stateless activation, so it composes inside `Sequential`.

    Deliberately opens no scope: it adds no parameters and one op, so a level
    here would only deepen every path.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def forward(self, x: TensorRef) -> TensorRef:
        return builder_of(x).unary(self.kind, x)


def _shape_of(data: WeightData) -> tuple[int, ...]:
    from ..builder import _infer_shape

    return _infer_shape(data)
