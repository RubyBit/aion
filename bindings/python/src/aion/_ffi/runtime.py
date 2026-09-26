# SPDX-License-Identifier: Apache-2.0
"""Typed façade for context, model, and tensor C ABI operations."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from ..device import GpuOptions, normalize_gpu_backend, normalize_gpu_power
from ..enums import AionDType
from ._raw import ffi, lib
from .dlpack import HostView
from .handles import ContextHandle, ModelHandle, TensorHandle
from .status import raise_for_status


def create_context(thread_count: int, gpus: Sequence[GpuOptions]) -> ContextHandle:
    count = len(gpus)
    c_gpus = ffi.NULL if count == 0 else ffi.new("AionGpuOptions[]", count)
    for i, gpu in enumerate(gpus):
        if not isinstance(gpu, GpuOptions):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"gpus[{i}] must be a GpuOptions, got {type(gpu).__name__}")
        c_gpus[i].power = normalize_gpu_power(gpu.power)
        c_gpus[i].backend = normalize_gpu_backend(gpu.backend)
        c_gpus[i].adapter_index = -1 if gpu.adapter_index is None else int(gpu.adapter_index)
    out = ffi.new("AionContext**")
    status = lib.aion_context_create(int(thread_count), c_gpus, count, out)
    raise_for_status(status, None, what="aion_context_create")
    return ContextHandle(out[0])


def destroy_context(ctx: ContextHandle) -> None:
    lib.aion_context_destroy(ctx.raw)


def load_model(
    ctx: ContextHandle,
    path: str,
    *,
    absolute: bool,
    device_kind: int | None,
    device_index: int,
    cache_capacity: int | None,
    growable: bool,
    initial_cache_capacity: int,
    auto_positions: bool,
) -> ModelHandle:
    out = ffi.new("AionLoadedModel**")
    c_path = ffi.new("char[]", path.encode("utf-8"))
    opts = ffi.new("AionLoadModelOptions*")
    opts.auto_init_inputs = 1
    opts.auto_positions = 1 if auto_positions else 0
    if device_kind is not None:
        opts.device_kind = int(device_kind)
        opts.device_index = int(device_index)
    if cache_capacity is not None and int(cache_capacity) > 0:
        opts.cache_capacity_tokens = int(cache_capacity)
        if growable:
            opts.cache_growable = 1
            opts.cache_initial_capacity_tokens = int(initial_cache_capacity)
    if absolute:
        status = lib.aion_loaded_model_load_path_absolute(ctx.raw, c_path, opts, out)
        what = "aion_loaded_model_load_path_absolute"
    else:
        status = lib.aion_loaded_model_load_path(ctx.raw, c_path, opts, out)
        what = "aion_loaded_model_load_path"
    raise_for_status(status, ctx, what=what)
    return ModelHandle(out[0])


def destroy_model(model: ModelHandle) -> None:
    lib.aion_loaded_model_destroy(model.raw)


def model_count(model: ModelHandle, kind: Literal["input", "output"]) -> int:
    fn = lib.aion_loaded_model_input_count if kind == "input" else lib.aion_loaded_model_output_count
    return int(fn(model.raw))


def model_name(
    ctx: ContextHandle,
    model: ModelHandle,
    kind: Literal["input", "output"],
    index: int,
) -> str:
    fn = lib.aion_loaded_model_input_name if kind == "input" else lib.aion_loaded_model_output_name
    out_len = ffi.new("size_t*")
    status = fn(model.raw, int(index), ffi.NULL, 0, out_len)
    raise_for_status(status, ctx, what=f"aion_loaded_model_{kind}_name")
    size = int(out_len[0])
    if size <= 0:
        return ""
    buf = ffi.new("char[]", size + 1)
    status = fn(model.raw, int(index), buf, size + 1, out_len)
    raise_for_status(status, ctx, what=f"aion_loaded_model_{kind}_name")
    return ffi.string(buf).decode("utf-8", errors="replace")


def model_dtype(
    ctx: ContextHandle,
    model: ModelHandle,
    kind: Literal["input", "output"],
    index: int,
) -> AionDType:
    fn = lib.aion_loaded_model_input_dtype if kind == "input" else lib.aion_loaded_model_output_dtype
    out = ffi.new("AionDType*")
    status = fn(model.raw, int(index), out)
    raise_for_status(status, ctx, what=f"aion_loaded_model_{kind}_dtype")
    return AionDType(int(out[0]))


def model_rank(
    ctx: ContextHandle,
    model: ModelHandle,
    kind: Literal["input", "output"],
    index: int,
) -> int:
    fn = lib.aion_loaded_model_input_rank if kind == "input" else lib.aion_loaded_model_output_rank
    out = ffi.new("size_t*")
    status = fn(model.raw, int(index), out)
    raise_for_status(status, ctx, what=f"aion_loaded_model_{kind}_rank")
    return int(out[0])


def bind_model_input(
    ctx: ContextHandle, model: ModelHandle, name: str, tensor: TensorHandle
) -> None:
    status = lib.aion_loaded_model_bind_input(
        model.raw, ffi.new("char[]", name.encode("utf-8")), tensor.raw
    )
    raise_for_status(status, ctx, what="aion_loaded_model_bind_input")


def set_state_input_policy(
    ctx: ContextHandle,
    model: ModelHandle,
    name: str,
    *,
    kind: Literal["growable", "rolling"],
    initial_capacity: int = 0,
    max_capacity: int = 0,
    growth_numerator: int = 0,
    growth_denominator: int = 0,
    retained_history: int = 0,
) -> None:
    policy = 1 if kind == "growable" else 2
    status = lib.aion_loaded_model_set_state_input_policy(
        model.raw,
        ffi.new("char[]", name.encode("utf-8")),
        policy,
        int(initial_capacity),
        int(growth_numerator),
        int(growth_denominator),
        int(max_capacity),
        int(retained_history),
    )
    raise_for_status(status, ctx, what="aion_loaded_model_set_state_input_policy")


def model_output_is_state(ctx: ContextHandle, model: ModelHandle, index: int) -> bool:
    out = ffi.new("uint8_t*")
    status = lib.aion_loaded_model_output_is_state(model.raw, int(index), out)
    raise_for_status(status, ctx, what="aion_loaded_model_output_is_state")
    return bool(out[0])


def model_position(ctx: ContextHandle, model: ModelHandle) -> int:
    out = ffi.new("uint64_t*")
    status = lib.aion_loaded_model_position(model.raw, out)
    raise_for_status(status, ctx, what="aion_loaded_model_position")
    return int(out[0])


def set_model_position(ctx: ContextHandle, model: ModelHandle, tokens: int) -> None:
    status = lib.aion_loaded_model_set_position(model.raw, int(tokens))
    raise_for_status(status, ctx, what="aion_loaded_model_set_position")


def run_model(ctx: ContextHandle, model: ModelHandle) -> None:
    status = lib.aion_loaded_model_run(model.raw)
    raise_for_status(status, ctx, what="aion_loaded_model_run")


def reset_model_state(ctx: ContextHandle, model: ModelHandle) -> None:
    status = lib.aion_loaded_model_reset_state(model.raw)
    raise_for_status(status, ctx, what="aion_loaded_model_reset_state")


def model_output_tensor(
    ctx: ContextHandle, model: ModelHandle, name: str
) -> TensorHandle:
    out = ffi.new("AionTensor**")
    status = lib.aion_loaded_model_output_tensor(
        model.raw, ffi.new("char[]", name.encode("utf-8")), out
    )
    raise_for_status(status, ctx, what="aion_loaded_model_output_tensor")
    return TensorHandle(out[0])


def create_empty_tensor(
    ctx: ContextHandle, dtype: AionDType, shape: Sequence[int]
) -> TensorHandle:
    dims = tuple(int(dim) for dim in shape)
    c_shape = ffi.NULL if not dims else ffi.new("size_t[]", dims)
    out = ffi.new("AionTensor**")
    status = lib.aion_tensor_create(ctx.raw, int(dtype), len(dims), c_shape, out)
    raise_for_status(status, ctx, what="aion_tensor_create")
    return TensorHandle(out[0])


def quantize_tensor(
    ctx: ContextHandle,
    dtype: AionDType,
    quant_axis: int,
    source: HostView,
) -> TensorHandle:
    """Quantize host memory `source` (any float dtype) into a tensor of its shape."""
    out = ffi.new("AionTensor**")
    status = lib.aion_tensor_quantize(ctx.raw, int(dtype), int(quant_axis), source.ptr, out)
    raise_for_status(status, ctx, what="aion_tensor_quantize")
    return TensorHandle(out[0])


def destroy_tensor(tensor: TensorHandle) -> None:
    lib.aion_tensor_destroy(tensor.raw)


def move_tensor(
    ctx: ContextHandle, tensor: TensorHandle, device_kind: int, device_index: int
) -> None:
    status = lib.aion_tensor_to(tensor.raw, int(device_kind), int(device_index))
    raise_for_status(status, ctx, what="aion_tensor_to")


def tensor_device(ctx: ContextHandle, tensor: TensorHandle) -> tuple[int, int]:
    out_kind = ffi.new("AionDeviceKind*")
    out_index = ffi.new("uint32_t*")
    status = lib.aion_tensor_device(tensor.raw, out_kind, out_index)
    raise_for_status(status, ctx, what="aion_tensor_device")
    return int(out_kind[0]), int(out_index[0])


def tensor_dtype(tensor: TensorHandle) -> AionDType:
    return AionDType(int(lib.aion_tensor_dtype(tensor.raw)))


def tensor_shape(ctx: ContextHandle, tensor: TensorHandle) -> tuple[int, ...]:
    rank = int(lib.aion_tensor_rank(tensor.raw))
    if rank == 0:
        return ()
    dims = ffi.new("size_t[]", rank)
    status = lib.aion_tensor_shape(tensor.raw, dims, rank)
    raise_for_status(status, ctx, what="aion_tensor_shape")
    return tuple(int(dims[i]) for i in range(rank))


def write_view(ctx: ContextHandle, tensor: TensorHandle, source: HostView) -> None:
    """Copy host memory into `tensor` (same shape; the core converts within a kind)."""
    status = lib.aion_tensor_write(tensor.raw, source.ptr)
    raise_for_status(status, ctx, what="aion_tensor_write")


def read_view(ctx: ContextHandle, tensor: TensorHandle, target: HostView) -> None:
    """Copy `tensor` out into host memory of the same shape."""
    if target.read_only:
        raise ValueError("cannot read a tensor into read-only memory")
    status = lib.aion_tensor_read(tensor.raw, target.ptr)
    raise_for_status(status, ctx, what="aion_tensor_read")


def tensor_from_dlpack(ctx: ContextHandle, managed: Any, dtype: AionDType | None) -> TensorHandle:
    """A tensor of `managed`'s data (a `DLManagedTensorVersioned*` the core owns on
    success): its memory borrowed when it already is what the tensor needs, else
    converted in one copy. `dtype` None keeps the data's own."""
    c_dtype = ffi.NULL if dtype is None else ffi.new("AionDType*", int(dtype))
    out = ffi.new("AionTensor**")
    status = lib.aion_tensor_from_dlpack(ctx.raw, managed, c_dtype, out)
    raise_for_status(status, ctx, what="aion_tensor_from_dlpack")
    return TensorHandle(out[0])


def zero_tensor(ctx: ContextHandle, tensor: TensorHandle) -> None:
    status = lib.aion_tensor_zero(tensor.raw)
    raise_for_status(status, ctx, what="aion_tensor_zero")


def tensor_to_dlpack(ctx: ContextHandle, tensor: TensorHandle) -> Any:
    """The tensor's bytes as a read-only `DLManagedTensorVersioned*`, shared, not
    copied; whoever ends up owning it calls its deleter once."""
    out = ffi.new("DLManagedTensorVersioned**")
    status = lib.aion_tensor_to_dlpack(tensor.raw, out)
    raise_for_status(status, ctx, what="aion_tensor_to_dlpack")
    return out[0]


__all__ = [
    "bind_model_input",
    "create_context",
    "create_empty_tensor",
    "destroy_context",
    "destroy_model",
    "destroy_tensor",
    "load_model",
    "model_count",
    "model_dtype",
    "model_name",
    "model_output_is_state",
    "model_output_tensor",
    "model_position",
    "model_rank",
    "move_tensor",
    "quantize_tensor",
    "read_view",
    "reset_model_state",
    "run_model",
    "set_model_position",
    "set_state_input_policy",
    "tensor_device",
    "tensor_dtype",
    "tensor_from_dlpack",
    "tensor_shape",
    "tensor_to_dlpack",
    "write_view",
    "zero_tensor",
]
