# SPDX-License-Identifier: Apache-2.0
"""PyTorch-like layers for the tracing authoring path.

A layer holds its weights as `Parameter`s (raw data, no builder) and composes ops
in `forward`, which runs under the ambient builder set by `aion.compile`/`export`:

    class MLP(nn.Module):
        def __init__(self, w1, w2):
            self.fc1 = nn.Linear(w1)
            self.fc2 = nn.Linear(w2)
        def forward(self, x):
            return self.fc2(self.fc1(x).relu())

    model = aion.compile(MLP(w1, w2), aion.spec((None, 4)))

For manual, builder-explicit construction use `aion.Builder`/`aion.Value` directly.
"""
from __future__ import annotations

import weakref
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any, Optional, Sequence

from .._trace import active_builder
from ..builder import Builder, Value, WeightData
from ..dtype import normalize_dtype
from ..enums import AionDType


class Parameter:
    """A weight held as raw data, resolved to a graph param at trace time.

    `.value(builder)` binds the data as a param exactly once per builder (cached
    by builder identity), so re-compiling the same module under a new builder
    rebinds correctly and concurrent traces don't share value ids.
    """

    __slots__ = ("data", "dtype", "_cache")

    def __init__(self, data: WeightData, *, dtype: object = "f32") -> None:
        self.data: object = data
        self.dtype: AionDType = normalize_dtype(dtype)
        self._cache: weakref.WeakKeyDictionary[Builder, Value] = (
            weakref.WeakKeyDictionary()
        )

    @property
    def shape(self) -> tuple[int, ...]:
        from ..builder import _infer_shape

        return _infer_shape(self.data)

    def value(self, builder: Builder) -> Value:
        v = self._cache.get(builder)
        if v is None:
            v = builder.param(self.data, dtype=self.dtype)
            self._cache[builder] = v
        return v


class Module:
    """Base class: subclasses set `Parameter`/`Module` attributes and implement
    `forward`. Assigning them auto-registers into `_parameters`/`_modules` (no
    `super().__init__()` needed)."""

    def __setattr__(self, name: str, value: object) -> None:
        if isinstance(value, Parameter):
            self.__dict__.setdefault("_parameters", OrderedDict())[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault("_modules", OrderedDict())[name] = value
        object.__setattr__(self, name, value)

    def parameters(self) -> Iterator[Parameter]:
        """Yield this module's and submodules' Parameters, deduped by identity."""
        seen: set[int] = set()
        for p in self.__dict__.get("_parameters", {}).values():
            if id(p) not in seen:
                seen.add(id(p))
                yield p
        for m in self.__dict__.get("_modules", {}).values():
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    yield p

    def forward(
        self, *args: Value, **kwargs: Value
    ) -> Value:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, *args: Value, **kwargs: Value) -> Value:
        return self.forward(*args, **kwargs)


class Linear(Module):
    """`y = x @ W (+ b)`, with `W` in Aion matmul-B layout `[in, out]`.

    Pass ``dtype="q8_0"`` to quantize the weight in the core (bias stays float).
    """

    def __init__(
        self,
        weight: object,
        bias: object | None = None,
        *,
        dtype: object = "f32",
    ) -> None:
        self.weight: Parameter = Parameter(weight, dtype=dtype)
        self.bias: Parameter | None = (
            Parameter(bias, dtype="f32") if bias is not None else None
        )

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, x: Value
    ) -> Value:
        b = active_builder()
        y = b.matmul(x, self.weight.value(b))
        if self.bias is not None:
            y = b.broadcast_add(y, self.bias.value(b))
        return y


class RMSNorm(Module):
    """Root-mean-square layer norm over the last `len(normalized_shape)` dims."""

    def __init__(
        self,
        gamma: object,
        beta: object | None = None,
        *,
        eps: float = 1e-6,
        normalized_shape: Optional[Sequence[int]] = None,
    ) -> None:
        self.gamma: Parameter = Parameter(gamma, dtype="f32")
        # A zero beta is the common case; author it explicitly when omitted.
        self.beta: Parameter = Parameter(
            beta if beta is not None else _zeros_like(gamma), dtype="f32"
        )
        self.eps: float = float(eps)
        self.normalized_shape: list[int] = (
            list(normalized_shape) if normalized_shape is not None else list(_shape_of(gamma))
        )

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, x: Value
    ) -> Value:
        b = active_builder()
        return b.rmsnorm(
            x, self.gamma.value(b), self.beta.value(b),
            eps=self.eps, normalized_shape=self.normalized_shape,
        )


def _shape_of(data: object) -> tuple[int, ...]:
    from ..builder import _infer_shape

    return _infer_shape(data)


def _zeros_like(data: object) -> object:
    try:
        import numpy as np
    except ImportError:
        np = None
    shape = _shape_of(data)
    if np is not None:
        return np.zeros(shape, dtype=np.float32)

    def build(dims: list[int]) -> object:
        if len(dims) == 1:
            return [0.0] * dims[0]
        return [build(dims[1:]) for _ in range(dims[0])]

    return build(list(shape)) if shape else 0.0


__all__ = ["Module", "Parameter", "Linear", "RMSNorm"]
