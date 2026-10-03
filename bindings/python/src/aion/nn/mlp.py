# SPDX-License-Identifier: Apache-2.0
"""Feed-forward blocks, mirroring `src/aion/api/nn/mlp.zig`."""
from __future__ import annotations

from typing import Optional

from ..dtype import float32
from ..types import DTypeLike
from ..builder import TensorRef, WeightData
from .layers import Linear
from .module import Module, builder_of


class FeedForward(Module):
    """`w2(act(w1(x)))` — the plain two-matmul feed-forward block."""

    def __init__(
        self,
        w1: WeightData,
        w2: WeightData,
        *,
        act: str = "silu",
        name: Optional[str] = None,
        bias1: Optional[WeightData] = None,
        bias2: Optional[WeightData] = None,
        dtype: DTypeLike = float32,
    ) -> None:
        self._layer_name = name
        self.act = act
        self.fc1 = Linear(w1, bias1, dtype=dtype)
        self.fc2 = Linear(w2, bias2, dtype=dtype)

    def forward(self, x: TensorRef) -> TensorRef:
        b = builder_of(x)
        with self._scoped(b):
            return self.fc2(b.unary(self.act, self.fc1(x)))


class GatedMLP(Module):
    """`down(gate(act, gate_proj(x), up(x)))` — SwiGLU (`silu`) / GeGLU (`gelu`).

    `gate` and `up` are two separate projections, the way every checkpoint ships
    them.

    The gate is one `gate` op whatever the activation, so the graph records the
    gated unit the author meant instead of a unary and a multiply for a compiler
    pass to recognise.
    """

    def __init__(
        self,
        gate: WeightData,
        up: WeightData,
        down: WeightData,
        *,
        act: str = "silu",
        name: Optional[str] = None,
        dtype: DTypeLike = float32,
    ) -> None:
        self._layer_name = name
        self.act = act

        ffn = _out_features(gate)
        if _out_features(up) != ffn:
            raise ValueError("gate and up must have the same output width")

        self.gate_proj = Linear(gate, dtype=dtype)
        self.up_proj = Linear(up, dtype=dtype)
        self.down_proj = Linear(down, dtype=dtype)

    def forward(self, x: TensorRef) -> TensorRef:
        b = builder_of(x)
        with self._scoped(b):
            g = self.gate_proj(x)
            u = self.up_proj(x)
            # The activation and the multiply a gated FFN is; the compiler fuses
            # them into one kernel where a fused kernel applies.
            gated = b.mul(b.unary(self.act, g), u)
            return self.down_proj(gated)


class GLU(Module):
    """`a * sigmoid(b)` over a projection split in half along the last dim."""

    def __init__(
        self,
        weight: WeightData,
        bias: Optional[WeightData] = None,
        *,
        name: Optional[str] = None,
        dtype: DTypeLike = float32,
    ) -> None:
        self._layer_name = name
        total = _out_features(weight)
        if total % 2 != 0:
            raise ValueError(f"GLU projection width must be even, got {total}")
        self.half = total // 2
        self.proj = Linear(weight, bias, dtype=dtype)

    def forward(self, x: TensorRef) -> TensorRef:
        b = builder_of(x)
        with self._scoped(b):
            both = self.proj(x)
            a = b.slice_last_dim(both, 0, self.half)
            g = b.slice_last_dim(both, self.half, self.half)
            return b.mul(b.unary("sigmoid", g), a)


def _out_features(data: WeightData) -> int:
    """A `Linear` weight's output width: the rows of its `[out, in]`."""
    from .._ffi.dlpack import data_shape

    shape = data_shape(data)
    if len(shape) != 2:
        raise ValueError(f"a Linear weight is [out, in], got shape {tuple(shape)}")
    return int(shape[0])
