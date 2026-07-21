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
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

from .dtype import normalize_dtype
from .enums import AionDType

# The builder active during a trace. Set only by compile/export.
_ACTIVE: contextvars.ContextVar = contextvars.ContextVar("aion_active_builder", default=None)


def active_builder():
    """The builder for the in-progress trace; raises if called outside one."""
    b = _ACTIVE.get()
    if b is None:
        raise RuntimeError(
            "nn layers must be called inside aion.compile()/aion.export(); "
            "for manual graph construction use aion.Builder directly."
        )
    return b


@contextlib.contextmanager
def _tracing(builder):
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


def spec(shape: Sequence[Optional[int]], dtype="f32", name: Optional[str] = None,
         *, placeholder: int = 1) -> InputSpec:
    """Describe a traced input. `None` in `shape` = a dynamic axis."""
    shp = tuple(shape)
    for d in shp:
        if d is not None and (not isinstance(d, int) or isinstance(d, bool) or d <= 0):
            raise ValueError(f"spec shape dims must be positive ints or None, got {d!r}")
    if placeholder <= 0:
        raise ValueError("placeholder must be > 0")
    return InputSpec(shape=shp, dtype=normalize_dtype(dtype), name=name, placeholder=int(placeholder))


def _resolve_input_names(module, specs: Sequence[InputSpec]) -> list[str]:
    """spec.name > forward parameter name > input{i}."""
    params: list[str] = []
    try:
        sig = inspect.signature(module.forward)
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


def _trace_module(module, specs: Sequence[InputSpec], *, ctx=None):
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
            outputs = module(*inputs)
        return builder, outputs
    except BaseException:
        builder.close()
        raise


def compile(module, *specs: InputSpec, device=None, ctx=None):
    """Trace `module` and compile it to an in-process `LoadedModel`.

    The traced builder owns the weight tensors the compiled model references, so
    it is attached to the model and closed with it.
    """
    builder, outputs = _trace_module(module, specs, ctx=ctx)
    try:
        model = builder.compile(outputs, device=device)
    except BaseException:
        builder.close()
        raise
    model._attach_authoring_builder(builder)
    return model


def export(module, *args, path=None, ctx=None) -> None:
    """Trace `module` and serialize it to a `.aion` file.

    Accepts the path either positionally (`export(m, spec, "m.aion")`) or as
    `path=`. The builder is closed once bytes are written.
    """
    specs = list(args)
    if path is None:
        if specs and isinstance(specs[-1], (str, os.PathLike)):
            path = specs.pop()
        else:
            raise TypeError("export() requires a path (as the last positional arg or path=)")
    builder, outputs = _trace_module(module, specs, ctx=ctx)
    try:
        builder.export(str(path), outputs)
    finally:
        builder.close()


__all__ = ["InputSpec", "spec", "compile", "export", "active_builder"]
