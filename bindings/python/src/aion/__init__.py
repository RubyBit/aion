# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from .context import Context, get_default_context, reset_default_context
from .device import DeviceLike, GpuOptions, _gpu_adapter_from_device
from .errors import AionError
from .model import LoadedModel, TensorSpec
from .tensor import Tensor
from .enums import AionDeviceKind, AionDType, AionGpuBackend, AionGpuPower, AionStatus

__all__ = [
    "AionDType",
    "AionDeviceKind",
    "AionError",
    "AionGpuBackend",
    "AionGpuPower",
    "AionStatus",
    "Context",
    "GpuOptions",
    "LoadedModel",
    "TensorSpec",
    "Tensor",
    "load_model",
    "reset_default_context",
]

# Python package version (may be independent of the core runtime version).
__version__ = "0.0.1"


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
