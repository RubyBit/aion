# SPDX-License-Identifier: Apache-2.0
"""Tracing front-end: turn an `nn.Module` into a runnable/serialized graph.

`compile`/`export` run the module's `forward` once under a scoped *ambient
builder* (a `contextvars` slot), so layers add ops to that builder without any
builder threaded through user code. The ambient scope is private — the explicit
escape hatch for manual graph construction is `aion.Builder` directly.
"""
from __future__ import annotations

import contextlib
import contextvars
import inspect
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence, Tuple, cast

from .builder import Builder, OutputsLike, Value
from .context import Context
from .device import DeviceLike
from .dtype import normalize_dtype
from .enums import AionDType
from .model import LoadedModel
from .tensor import Tensor

# The builder active during a trace. Set only by compile/export.
_ACTIVE: contextvars.ContextVar[Builder | None] = contextvars.ContextVar(
    "aion_active_builder", default=None
)


def active_builder() -> Builder:
    """The builder for the in-progress trace; raises if called outside one."""
    b = _ACTIVE.get()
    if b is None:
        raise RuntimeError(
            "nn layers must be called inside aion.compile()/aion.export(); "
            "for manual graph construction use aion.Builder directly."
        )
    return b


@contextlib.contextmanager
def _tracing(builder: Builder) -> Iterator[Builder]:
    token = _ACTIVE.set(builder)
    try:
        yield builder
    finally:
        _ACTIVE.reset(token)


@dataclass(frozen=True)
class InputSpec:
    """A traced input: `shape` with `None` marking dynamic (runtime-varying) axes.

    `placeholder` is the concrete authoring size used for each `None` axis during
    the trace (eager shape inference needs a concrete int); the compiled/exported
    model still serves any size on those axes.
    """

    shape: Tuple[Optional[int], ...]
    dtype: AionDType
    name: Optional[str] = None
    placeholder: int = 1


def spec(shape: Sequence[Optional[int]], dtype: object = "f32", name: Optional[str] = None,
         *, placeholder: int = 1) -> InputSpec:
    """Describe a traced input. `None` in `shape` = a dynamic axis."""
    shp = tuple(shape)
    for d in shp:
        if d is not None and (not isinstance(d, int) or isinstance(d, bool) or d <= 0):
            raise ValueError(f"spec shape dims must be positive ints or None, got {d!r}")
    if placeholder <= 0:
        raise ValueError("placeholder must be > 0")
    return InputSpec(shape=shp, dtype=normalize_dtype(dtype), name=name, placeholder=int(placeholder))


def _resolve_input_names(module: object, specs: Sequence[InputSpec]) -> list[str]:
    """spec.name > forward parameter name > input{i}."""
    params: list[str] = []
    try:
        sig = inspect.signature(cast(Any, module).forward)
        params = [
            p.name for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        params = []
    names: list[str] = []
    for i, s in enumerate(specs):
        if s.name:
            names.append(s.name)
        elif i < len(params):
            names.append(params[i])
        else:
            names.append(f"input{i}")
    return names


def _trace_module(
    module: object,
    specs: Sequence[InputSpec],
    *,
    ctx: Context | None = None,
) -> tuple[Builder, OutputsLike]:
    """Build inputs from specs, run forward once under the ambient builder.

    Returns `(builder, outputs)` where `outputs` is whatever `forward` returned
    (a `Value`, a sequence, or a `{name: Value}` mapping). The builder is left
    open; the caller owns tearing it down.
    """
    from .builder import Builder

    if not specs:
        raise ValueError("compile()/export() require at least one input spec")

    builder = Builder(ctx=ctx)
    try:
        names = _resolve_input_names(module, specs)
        inputs = []
        for s, nm in zip(specs, names):
            dims = [s.placeholder if d is None else int(d) for d in s.shape]
            dyn = [i for i, d in enumerate(s.shape) if d is None] or None
            v = builder.input(dims, dtype=s.dtype, dynamic=dyn)
            builder.name(v, nm)
            inputs.append(v)
        with _tracing(builder):
            outputs = cast(Any, module)(*inputs)
        return builder, cast(OutputsLike, outputs)
    except BaseException:
        builder.close()
        raise


def _is_graph_outputs(obj: object) -> bool:
    """True if `obj` is a lazy `Tensor` (or a list/dict of them) — the
    Tensor-first authoring path, as opposed to an `nn.Module` to trace."""
    def is_lazy(v: object) -> bool:
        return isinstance(v, Tensor) and v._lazy

    if is_lazy(obj):
        return True
    if isinstance(obj, Mapping):
        vals = list(obj.values())
        return bool(vals) and all(is_lazy(v) for v in vals)
    if isinstance(obj, (list, tuple)):
        return bool(obj) and all(is_lazy(v) for v in obj)
    return False


def _named_outputs(outputs: object) -> list[tuple[str, Tensor]]:
    """Normalize `outputs` to an ordered list of `(name, lazy_tensor)`.

    A `{name: tensor}` mapping names explicitly; otherwise a tensor's own
    `.rename(...)` is the default, falling back to positional `output{i}`.
    """
    if isinstance(outputs, Mapping):
        return [
            (str(key), cast(Tensor, value))
            for key, value in outputs.items()
        ]
    seq = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
    typed = [cast(Tensor, value) for value in seq]
    return [(value._name or f"output{i}", value) for i, value in enumerate(typed)]


def _compile_graph(
    outputs: object, *, device: DeviceLike = None
) -> LoadedModel:
    """Lower a lazy-Tensor DAG into a fresh builder and compile it in place."""
    from .tensor import _lower

    named = _named_outputs(outputs)
    b, out_values = _lower([t for _, t in named])
    for (nm, _), v in zip(named, out_values):
        b.mark_output(v, nm)
    model = b.compile(device=device)
    model._attach_authoring_builder(b)
    return model


def compile(
    module: object,
    *specs: InputSpec,
    inputs: Sequence[Tensor] | None = None,
    device: DeviceLike = None,
    ctx: Context | None = None,
) -> LoadedModel:
    """Compile a model to an in-process `LoadedModel`.

    Two forms:
    - **Tensor-first:** ``aion.compile(outputs, inputs=[...])`` where `outputs`
      is a lazy `Tensor` / list / ``{name: Tensor}`` built with operators, and
      `inputs` are the `aion.tensor(shape=...)` free inputs to expose.
    - **nn tracing:** ``aion.compile(module, spec(...), ...)`` traces an
      `nn.Module`'s `forward` once.

    The authoring builder owns the model's weight tensors, so it is attached to
    the model and closed with it.
    """
    if _is_graph_outputs(module):
        return _compile_graph(module, device=device)

    builder, outputs = _trace_module(module, specs, ctx=ctx)
    try:
        model = builder.compile(outputs, device=device)
    except BaseException:
        builder.close()
        raise
    model._attach_authoring_builder(builder)
    return model


def export(
    module: object,
    *args: object,
    path: str | os.PathLike[str] | None = None,
    inputs: Sequence[Tensor] | None = None,
    ctx: Context | None = None,
) -> None:
    """Serialize a model to a `.aion` file.

    Two forms (mirroring `compile`):
    - **Tensor-first:** ``aion.export(outputs, "m.aion", inputs=[...])``.
    - **nn tracing:** ``aion.export(module, spec(...), "m.aion")``.

    The path is the last positional arg or `path=`.
    """
    if _is_graph_outputs(module):
        from .tensor import _lower

        p = path
        rest = list(args)
        if p is None:
            if rest and isinstance(rest[-1], (str, os.PathLike)):
                p = rest.pop()
            else:
                raise TypeError("export() requires a path (as the last positional arg or path=)")
        named = _named_outputs(module)
        b, out_values = _lower([t for _, t in named])
        try:
            for (nm, _), v in zip(named, out_values):
                b.mark_output(v, nm)
            b.export(str(p))
        finally:
            b.close()
        return

    specs = list(args)
    if path is None:
        if specs and isinstance(specs[-1], (str, os.PathLike)):
            path = cast(str | os.PathLike[str], specs.pop())
        else:
            raise TypeError("export() requires a path (as the last positional arg or path=)")
    builder, outputs = _trace_module(
        module, cast(list[InputSpec], specs), ctx=ctx
    )
    try:
        builder.export(str(path), outputs)
    finally:
        builder.close()


__all__ = ["InputSpec", "spec", "compile", "export", "active_builder"]
