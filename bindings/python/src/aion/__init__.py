# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from . import nn
from .builder import Builder, DynamicAxes, Value
from .context import Context, get_default_context, reset_default_context
from .device import DeviceLike, GpuOptions, _gpu_adapter_from_device
from .dtype import float16, float32, int8, int32, normalize_dtype, q4_0, q8_0
from .errors import AionError
from .model import LoadedModel, TensorSpec
from .tensor import Tensor
from ._trace import InputSpec, compile, export, spec
from .enums import AionDeviceKind, AionDType, AionGpuBackend, AionGpuPower, AionStatus

__all__ = [
    "AionDType",
    "AionDeviceKind",
    "AionError",
    "AionGpuBackend",
    "AionGpuPower",
    "AionStatus",
    "Builder",
    "Context",
    "GpuOptions",
    "InputSpec",
    "LoadedModel",
    "TensorSpec",
    "Tensor",
    "Value",
    "compile",
    "export",
    "float16",
    "float32",
    "int8",
    "int32",
    "load_model",
    "nn",
    "normalize_dtype",
    "q4_0",
    "q8_0",
    "realize",
    "reset_default_context",
    "spec",
    "tensor",
]

# Python package version (may be independent of the core runtime version).
__version__ = "0.0.1"


def tensor(
    data: object | None = None,
    *,
    shape: Sequence[int] | None = None,
    dtype: object | None = None,
    device: DeviceLike = None,
    name: str | None = None,
    dynamic: DynamicAxes = None,
    ctx: Context | None = None,
) -> Tensor:
    """Create a tensor — the Tensor-first front door.

    - ``aion.tensor(data)`` — a **concrete** tensor from a numpy array, nested
      list, or scalar. The dtype is inferred (integer data → i32, else
      f32) unless you pass ``dtype=``. ``device=`` migrates it after creation.
    - ``aion.tensor(shape=(...))`` — a **graph** free input (symbolic): the
      starting point for authoring a model, wired with operators and then
      compiled via ``aion.compile(outputs, inputs=[...])`` or exported. Pass
      ``name=`` to bind it by name at runtime, and ``dynamic=`` to mark
      runtime-varying axes (see `Builder.input`).

    Pass exactly one of ``data`` (positional) or ``shape=``.
    """
    if shape is not None:
        if data is not None:
            raise TypeError("aion.tensor: pass positional data or shape=, not both")
        if device is not None:
            raise TypeError("aion.tensor: device= is not valid for a graph input (shape=)")
        rctx = ctx if ctx is not None else get_default_context()
        attrs = {"shape": tuple(int(d) for d in shape),
                 "dtype": ("f32" if dtype is None else dtype),
                 "dynamic": dynamic}
        return Tensor._from_node("input", [], attrs, ctx=rctx, name=name)
    if data is not None and name is not None:
        raise TypeError("aion.tensor: name= is only valid for a graph input (shape=)")
    if data is None:
        raise TypeError("aion.tensor: provide positional data or shape=")
    return Tensor(data, ctx=ctx, dtype=dtype, device=device)


def realize(
    *tensors: Tensor | Sequence[Tensor],
    device: DeviceLike = None,
) -> list[Tensor]:
    """Materialize one or more lazy tensors in a single compile + run.

    Accepts ``aion.realize(a, b)`` or ``aion.realize([a, b])``. Shared subgraphs
    are emitted once, so passing several results computes them together. Returns
    the list of the now-concrete tensors.
    """
    from .tensor import _realize_many

    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        group = list(tensors[0])
    else:
        group = [cast(Tensor, value) for value in tensors]
    _realize_many(group, device=device)
    return group


def load_model(
    path: str,
    *,
    thread_count: int | None = None,
    device: DeviceLike = None,
    adapter_index: int | None = None,
    power: str = "high",
    backend: str = "any",
    cache_capacity: int | None = None,
    growable: bool = False,
    initial_cache_capacity: int = 8,
    auto_positions: bool = True,
) -> LoadedModel:
    """Convenience helper for loading models.

    - If `device` is None/`"cpu"`: CPU. With `thread_count` omitted, the
      process-wide default Context is used; otherwise a dedicated Context.
    - If `device` selects a GPU, a dedicated single-GPU Context is created and the
      model loaded onto it. Because that context registers exactly one GPU, the
      device index names the **physical adapter**:
        * ``device="gpu"``      -> pick by `power` (default ``"high"`` = discrete)
        * ``device="gpu:1"``    -> physical adapter 1 (integrated/discrete order
          is what wgpu enumerates; use to pick between them)
        * ``adapter_index=N``   -> equivalent explicit form
      For multiple GPUs registered on one context, build `Context(gpus=[...])`
      yourself and use `LoadedModel.load(ctx, path, device="gpu:i")` (registry index).
    - `cache_capacity` / `growable` / `initial_cache_capacity` / `auto_positions`
      configure role-declared state (see `LoadedModel.load`): the runtime
      allocates/zeros KV-style caches itself and auto-feeds cache index /
      position inputs, so a stateful model needs only its real inputs bound.
    """

    def _load(ctx: Context, dev: DeviceLike = None) -> LoadedModel:
        return LoadedModel.load(
            ctx,
            path,
            device=dev,
            cache_capacity=cache_capacity,
            growable=growable,
            initial_cache_capacity=initial_cache_capacity,
            auto_positions=auto_positions,
        )

    is_gpu, dev_idx = _gpu_adapter_from_device(device)
    if not is_gpu:
        if thread_count is None:
            return _load(get_default_context())
        ctx = Context(thread_count=thread_count)
        try:
            return _load(ctx)
        except Exception:
            ctx.close()
            raise

    if adapter_index is not None and dev_idx is not None and adapter_index != dev_idx:
        raise ValueError(
            f"conflicting GPU index: device={device!r} implies adapter {dev_idx}, "
            f"but adapter_index={adapter_index}"
        )
    resolved_adapter = adapter_index if adapter_index is not None else dev_idx

    ctx = Context.gpu(thread_count=thread_count, adapter_index=resolved_adapter, power=power, backend=backend)
    try:
        return _load(ctx, "gpu:0")
    except Exception:
        ctx.close()
        raise
