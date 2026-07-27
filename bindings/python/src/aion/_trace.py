# SPDX-License-Identifier: Apache-2.0
"""Tracing front-end: turn an `nn.Module` into a runnable/serialized graph.

`compile`/`export` build the declared inputs on a fresh `Builder` and run the
module's `forward` once. No ambient state is involved: a layer reads the builder
off the `TensorRef` it is given, so the same layer works here and against a bare
`aion.Builder`.
"""
from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, cast

from .builder import Builder, OutputsLike, TensorRef
from .context import Context
from .device import DeviceLike
from .dtype import normalize_dtype
from .enums import AionDType
from .model import LoadedModel
from .tensor import Tensor

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
    (a `TensorRef`, a sequence, or a `{name: TensorRef}` mapping). The builder is left
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
        outputs = cast(Any, module)(*inputs)
        return builder, cast(OutputsLike, outputs)
    except BaseException:
        builder.close()
        raise


def compile(
    module: object,
    *specs: InputSpec,
    device: DeviceLike = None,
    ctx: Context | None = None,
) -> LoadedModel:
    """Trace `module`'s `forward` once and compile it to a `LoadedModel`.

    ``aion.compile(module, spec(...), ...)``. Inputs are built from the specs and
    passed to `forward` as `TensorRef`s, so a module composes the same ops a raw
    `Builder` would — there is one graph representation.

    The authoring builder owns the model's weight tensors, so it is attached to
    the model and closed with it.
    """
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
    ctx: Context | None = None,
) -> None:
    """Serialize a traced module to a `.aion` file.

    ``aion.export(module, spec(...), "m.aion")``. The path is the last positional
    arg or `path=`.
    """
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


__all__ = ["InputSpec", "spec", "compile", "export"]
