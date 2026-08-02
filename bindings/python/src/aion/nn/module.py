# SPDX-License-Identifier: Apache-2.0
"""Module/Parameter infrastructure for the Python layer catalog.

Mirrors `src/aion/api/nn` in shape. The stateful machinery — scopes, parameter
naming, constant caching — lives once in the Zig `Builder` and is reached through
the C ABI; what is written here is the per-layer composition.

A layer takes and returns `TensorRef`, and derives the builder from its input, so
there is no ambient state: `layer(x)` works wherever `x` came from.
"""
from __future__ import annotations

import contextlib
import weakref
from collections import OrderedDict
from collections.abc import Callable, Generator, Iterator
from typing import TYPE_CHECKING, Any, Generic, Optional

from ..builder import Builder, TensorRef, WeightData
from ..dtype import float32
from ..types import DTypeLike, Shape


def builder_of(x: TensorRef) -> Builder:
    """The builder a value belongs to.

    Layers read this instead of consulting ambient state, which is what lets the
    same layer be used inside `aion.compile(...)` and against a bare `Builder`.
    """
    b = getattr(x, "_b", None)
    if b is None:
        raise TypeError(
            f"expected an aion.TensorRef produced by a Builder, got {type(x).__name__}"
        )
    return b


# What a layer's `forward` returns. Most produce one `TensorRef`, which is the
# default, so only the layers returning several name it —
# `LSTMCell(Module[tuple[TensorRef, TensorRef]])`. That keeps `layer(x)` precisely typed
# instead of `Any`, without a parameter on every declaration.
#
# PEP 696 defaults are not in 3.12's `typing`, and only the checker needs the
# default: at runtime a plain TypeVar is all `Generic` requires.
if TYPE_CHECKING:
    from typing_extensions import TypeVar

    _Out = TypeVar("_Out", default="TensorRef")
else:
    from typing import TypeVar

    _Out = TypeVar("_Out")


def _forward_unimplemented(self: "Module[Any]", *args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        f"{type(self).__name__} does not implement forward()"
    )


class Parameter:
    """A weight held as data, bound as a named graph param on first use.

    Binding is cached per builder, so re-compiling the same module under a new
    builder rebinds correctly and two concurrent graphs never share value ids.
    """

    __slots__ = ("data", "kwargs", "_cache")

    def __init__(
        self,
        data: WeightData,
        *,
        dtype: DTypeLike = float32,
        shape: Optional[Shape] = None,
        quant_axis: Optional[int] = None,
    ) -> None:
        self.data: WeightData = data
        # Passed through to `Builder.param_named`. `shape`/`quant_axis` matter for
        # quantized weights: an embedding table blocks along its feature axis, not
        # the matmul reduction axis.
        self.kwargs: dict[str, Any] = {"dtype": dtype}
        if shape is not None:
            self.kwargs["shape"] = shape
        if quant_axis is not None:
            self.kwargs["quant_axis"] = quant_axis
        self._cache: weakref.WeakKeyDictionary[Builder, TensorRef] = weakref.WeakKeyDictionary()

    @property
    def shape(self) -> tuple[int, ...]:
        from ..builder import _infer_shape

        return _infer_shape(self.data)

    def value(self, b: Builder, name: str) -> TensorRef:
        """Bind under `<current scope>/<name>`, or return the existing binding."""
        v = self._cache.get(b)
        if v is None:
            v = b.param_named(self.data, name, **self.kwargs)
            self._cache[b] = v
        return v

    def binding(self, b: Builder) -> Optional[TensorRef]:
        """The value this parameter was bound to under `b`, if it has been."""
        return self._cache.get(b)


class Module(Generic[_Out]):
    """Base class: set `Parameter`/`Module` attributes and implement `forward`.

    Assignment auto-registers into `_parameters`/`_modules` (no
    `super().__init__()` needed) and, for a submodule, names it after the
    attribute — so `self.fc1 = nn.Linear(w)` yields `fc1/weight`, deterministically
    and independent of construction order.
    """

    def __setattr__(self, name: str, value: object) -> None:
        if isinstance(value, Parameter):
            self.__dict__.setdefault("_parameters", OrderedDict())[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault("_modules", OrderedDict())[name] = value
            # The attribute name is the layer's identity unless it was named
            # explicitly, which is what keeps `state_dict` paths stable.
            if value._layer_name is None:
                value._layer_name = name
                value._on_named(name)
        object.__setattr__(self, name, value)

    def _on_named(self, name: str) -> None:
        """Hook: this module was just named `name` by its parent.

        Containers use it to push the name down. A container is never *called*,
        so its own segment would otherwise never reach the graph, and the parameter
        paths would disagree with the scopes the ops land in.
        """

    @property
    def _layer_name(self) -> Optional[str]:
        return self.__dict__.get("__layer_name")

    @_layer_name.setter
    def _layer_name(self, value: Optional[str]) -> None:
        self.__dict__["__layer_name"] = value

    # --- scope -------------------------------------------------------------

    @contextlib.contextmanager
    def _scoped(self, b: Builder) -> Generator[str, None, None]:
        """Enter this layer's scope, fixing its identity on first use.

        A named layer (the usual case, from its attribute name) enters the same
        scope every call. An unnamed root layer falls back to an auto-numbered
        segment, which is resolved once and then reused — otherwise a second
        `forward` would emit ops under a different path than its parameters.
        """
        explicit = self._layer_name
        if explicit is not None:
            with b.scope(explicit) as seg:
                yield seg
            return

        cache = self.__dict__.setdefault("__auto_segments", weakref.WeakKeyDictionary())
        seg = cache.get(b)
        if seg is None:
            with b.auto_scope(type(self).__name__) as resolved:
                cache[b] = resolved
                yield resolved
            return
        with b.scope(seg):
            yield seg

    # --- introspection ------------------------------------------------------

    def parameters(self) -> Iterator[Parameter]:
        """This module's and submodules' Parameters, deduped by identity."""
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

    def modules(self) -> Iterator["Module[Any]"]:
        yield self
        for m in self.__dict__.get("_modules", {}).values():
            yield from m.modules()

    def named_parameters(self, b: Builder) -> Iterator[tuple[str, Parameter]]:
        """Yield `(graph name, Parameter)` for parameters bound under `b`.

        Names come from the builder — the same strings that land in the package —
        rather than being recomputed from the attribute tree, so there is one
        answer to "where does this weight live". Mirrors `nn.forEachParam` in the
        Zig API, which takes the builder for the same reason.

        Only reports what has actually been bound, so call it after tracing
        `forward` once.
        """
        for p in self.parameters():
            v = p.binding(b)
            if v is None:
                continue
            name = b.value_name(v)
            if name:
                yield (name, p)

    def state_dict(self, b: Builder) -> "OrderedDict[str, object]":
        """`{graph name: data}` — the weights, keyed as the package stores them."""
        return OrderedDict((name, p.data) for name, p in self.named_parameters(b))

    def __repr__(self) -> str:
        subs = self.__dict__.get("_modules", {})
        head = type(self).__name__
        if not subs:
            return f"{head}()"
        body = "\n".join(f"  ({k}): {v!r}".replace("\n", "\n  ") for k, v in subs.items())
        return f"{head}(\n{body}\n)"

    # --- call ---------------------------------------------------------------
    #
    # `forward` is a class *attribute* of callable type rather than a declared
    # method. Layers have genuinely different signatures — `Linear(x)`,
    # `RoPE(x, positions)`, `LSTMCell(x, h, c) -> (h, c)` — and a declared base
    # method would make every one of them an "incompatible override". Each layer
    # still annotates its own `forward` precisely, so the signature is visible
    # where it is written.

    forward: Callable[..., _Out] = _forward_unimplemented

    def __call__(self, *args: Any, **kwargs: Any) -> _Out:
        return self.forward(*args, **kwargs)


class ModuleList(Module):
    """An ordered list of submodules, registered so they are reachable.

    Needed because a plain Python list is not registered — the same rule PyTorch
    has — and a transformer is a list of blocks. Children are named by index, so
    paths read `layers/3/attn/weight`.
    """

    def __init__(self, modules: "list[Module[Any]] | None" = None) -> None:
        self._items: list[Module[Any]] = []
        for m in modules or []:
            self.append(m)

    def append(self, m: "Module[Any]") -> "ModuleList":
        idx = len(self._items)
        self._items.append(m)
        # Register so `named_parameters` walks it, keyed by index.
        self.__dict__.setdefault("_modules", OrderedDict())[str(idx)] = m
        m._layer_name = self._child_name(idx)
        return self

    def _child_name(self, idx: int) -> str:
        own = self._layer_name
        return f"{own}/{idx}" if own else str(idx)

    def _on_named(self, name: str) -> None:
        # The list itself is iterated, never called, so its segment can only reach
        # the graph by living in each child's name.
        for idx, m in enumerate(self._items):
            m._layer_name = f"{name}/{idx}"

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator["Module[Any]"]:
        return iter(self._items)

    def __getitem__(self, i: int) -> "Module[Any]":
        return self._items[i]


class Sequential(Module):
    """Chain single-input/single-output layers in order.

    Children are named by position, so a `self.block = nn.Sequential(...)` yields
    `block/0/weight`, `block/1/weight`, ...
    """

    def __init__(self, *layers: "Module[Any]") -> None:
        self._items: list[Module[Any]] = list(layers)
        for i, m in enumerate(self._items):
            self.__dict__.setdefault("_modules", OrderedDict())[str(i)] = m
            m._layer_name = str(i)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator["Module[Any]"]:
        return iter(self._items)

    def forward(self, x: TensorRef) -> TensorRef:
        b = builder_of(x)
        with self._scoped(b):
            for layer in self._items:
                x = layer(x)
            return x


class Residual(Module):
    """`x + scale * body(x)`.

    `scale` covers the macaron-Conformer convention of adding half of a branch. It
    is bound as a shared one-element constant, so it costs one float for the whole
    model rather than a vector per use.
    """

    def __init__(self, body: "Module[TensorRef]", scale: float = 1.0) -> None:
        self.body = body
        self.scale = float(scale)

    def forward(self, x: TensorRef) -> TensorRef:
        b = builder_of(x)
        y = self.body(x)
        if self.scale != 1.0:
            y = b.mul(y, b.constant(self.scale))
        return b.add(x, y)
