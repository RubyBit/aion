# SPDX-License-Identifier: Apache-2.0
"""Stateless helpers that materialize a constant operand via the Builder.

Mirrors `src/aion/api/nn/functional.zig`. Everything else you might expect here
is already a `Builder`/`TensorRef` op — these exist only because Aion has no
scalar-typed operand.
"""
from __future__ import annotations

import math

from ..builder import TensorRef
from .module import builder_of


def scale(x: TensorRef, k: float) -> TensorRef:
    """`x * k`.

    The scalar is bound as a **one-element** operand to the broadcast op, so its
    cost is independent of `x`'s size, and the constant is shared across every use
    of the same `k`. Requires rank >= 2, which the broadcast op enforces.
    """
    if k == 1.0:
        return x
    b = builder_of(x)
    return b.mul(x, b.constant(k))


def shift(x: TensorRef, k: float) -> TensorRef:
    """`x + k`, same one-element-constant treatment as `scale`."""
    if k == 0.0:
        return x
    b = builder_of(x)
    return b.add(x, b.constant(k))


def softcap(x: TensorRef, cap: float) -> TensorRef:
    """`cap * tanh(x / cap)` — a smooth clamp to `(-cap, cap)`.

    Gemma applies this to attention logits and to the final logits. `cap <= 0`
    means disabled and returns `x` untouched, matching how the attention op reads
    its own soft-cap attribute.
    """
    if not cap > 0.0:
        return x
    if not math.isfinite(cap):
        raise ValueError("softcap must be finite")
    b = builder_of(x)
    return scale(b.unary("tanh", scale(x, 1.0 / cap)), cap)
