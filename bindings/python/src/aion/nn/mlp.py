# SPDX-License-Identifier: Apache-2.0
"""Feed-forward blocks, mirroring `src/aion/api/nn/mlp.zig`."""
from __future__ import annotations

from typing import Optional

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
        dtype: DTypeLike = "f32",
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
    """`down(act(gate(x)) * up(x))` — SwiGLU (`silu`) / GeGLU (`gelu`).

    `gate` and `up` are two separate projections, the way every checkpoint ships
    them. Pre-concatenating the pair into one `[in, 2*ffn]` weight is a *fusion*,
    not a layout, and it belongs to the compiler: `opt/fuse_horizontal_matmul`
    rewrites matmuls sharing an operand into one wide matmul plus a slice each,
    numerically identically. So there is nothing to choose here.

    The gelu case routes through the fused `gelu_mul` op rather than a separate
    unary and multiply.
    """

    def __init__(
        self,
        gate: WeightData,
        up: WeightData,
        down: WeightData,
        *,
        act: str = "silu",
        name: Optional[str] = None,
        dtype: DTypeLike = "f32",
    ) -> None:
        self._layer_name = name
        self.act = act

        ffn = int(_last_dim(gate))
        if int(_last_dim(up)) != ffn:
            raise ValueError("gate and up must have the same output width")

        self.gate_proj = Linear(gate, dtype=dtype)
        self.up_proj = Linear(up, dtype=dtype)
        self.down_proj = Linear(down, dtype=dtype)

    def forward(self, x: TensorRef) -> TensorRef:
        b = builder_of(x)
        with self._scoped(b):
            g = self.gate_proj(x)
            u = self.up_proj(x)
            gated = b.gelu_mul(g, u) if self.act == "gelu" else b.mul(b.unary(self.act, g), u)
            return self.down_proj(gated)


class GLU(Module):
    """`a * sigmoid(b)` over a projection split in half along the last dim."""

    def __init__(
        self,
        weight: WeightData,
        bias: Optional[WeightData] = None,
        *,
        name: Optional[str] = None,
        dtype: DTypeLike = "f32",
    ) -> None:
        self._layer_name = name
        total = int(_last_dim(weight))
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
            return b.mul(a, b.unary("sigmoid", g))


def _last_dim(data: object) -> int:
    from ..builder import _infer_shape

    shape = _infer_shape(data)
    if not shape:
        raise ValueError("weight must have at least one axis")
    return int(shape[-1])
