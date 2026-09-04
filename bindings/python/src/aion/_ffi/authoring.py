# SPDX-License-Identifier: Apache-2.0
"""Typed façade for the model-authoring portion of the C ABI."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, NewType, TypeAlias

from ..enums import AionDType, AionOp
from ..types import AttentionWindow
from ._raw import ffi, lib
from .handles import (
    BuilderHandle,
    ContextHandle,
    ModelHandle,
    TensorHandle,
)
from .status import raise_for_status

ValueId = NewType("ValueId", int)
RegionId = NewType("RegionId", int)


@dataclass(frozen=True)
class MatmulAttrs:
    alpha: float = 1.0
    beta: float = 0.0


@dataclass(frozen=True)
class ElemwiseAttrs:
    op: int


@dataclass(frozen=True)
class UnaryAttrs:
    op: int


@dataclass(frozen=True)
class AxisAttrs:
    axis: int


@dataclass(frozen=True)
class GatherAttrs:
    axis: int
    batch_dims: int = 0


@dataclass(frozen=True)
class NormAttrs:
    eps: float
    normalized_shape: tuple[int, ...]


@dataclass(frozen=True)
class Rope1DAttrs:
    base_frequency: float
    scale_factor: float
    rope_proportion: float


@dataclass(frozen=True)
class ViewDims:
    concrete: tuple[int, ...]
    symbols: tuple[str | None, ...]


@dataclass(frozen=True)
class ReshapeAttrs:
    dims: ViewDims


@dataclass(frozen=True)
class CastAttrs:
    dtype: AionDType


@dataclass(frozen=True)
class ReduceAttrs:
    op: int
    axis: int | None


@dataclass(frozen=True)
class Conv1DAttrs:
    stride: int
    dilation: int
    pad_left: int
    pad_right: int
    groups: int
    pad_mode: int


@dataclass(frozen=True)
class Conv2DAttrs:
    stride_h: int
    stride_w: int
    dilation_h: int
    dilation_w: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    groups: int
    pad_mode: int


@dataclass(frozen=True)
class StftAttrs:
    n_fft: int
    hop_length: int
    center: bool


@dataclass(frozen=True)
class OptionalAxisAttrs:
    axis: int | None


@dataclass(frozen=True)
class SliceAttrs:
    starts: tuple[int, ...]
    lens: ViewDims


def _set_window(member, w: AttentionWindow) -> None:
    member.left = int(w.left)
    member.right = int(w.right)
    member.chunk = int(w.chunk)


@dataclass(frozen=True)
class AttentionAttrs:
    scale: float
    window: AttentionWindow = AttentionWindow.CAUSAL
    attn_logits_soft_cap: float = 0.0
    has_query_positions: bool = False
    has_kv_lengths: bool = False


@dataclass(frozen=True)
class RelposMhaAttrs:
    #: No head count: it is q's dim 2 (q is ``[B, T, heads, head_dim]``).
    scale: float
    window: AttentionWindow = AttentionWindow.FULL
    relative_zero_index: int = 0
    attn_logits_soft_cap: float = 0.0


OpAttrs: TypeAlias = (
    MatmulAttrs
    | ElemwiseAttrs
    | UnaryAttrs
    | AxisAttrs
    | GatherAttrs
    | NormAttrs
    | Rope1DAttrs
    | ReshapeAttrs
    | CastAttrs
    | ReduceAttrs
    | Conv1DAttrs
    | Conv2DAttrs
    | StftAttrs
    | OptionalAxisAttrs
    | SliceAttrs
    | AttentionAttrs
    | RelposMhaAttrs
    | None
)


def _string(value: str):
    return ffi.new("char[]", value.encode("utf-8"))


def _view_dims(dims: ViewDims):
    concrete = ffi.new("size_t[]", dims.concrete)
    if not any(symbol is not None for symbol in dims.symbols):
        return concrete, ffi.NULL, [concrete]
    keepalive: list[object] = [concrete]
    pointers: list[object] = []
    for symbol in dims.symbols:
        if symbol is None:
            pointers.append(ffi.NULL)
        else:
            c_symbol = _string(symbol)
            keepalive.append(c_symbol)
            pointers.append(c_symbol)
    symbols = ffi.new("char*[]", pointers)
    keepalive.append(symbols)
    return concrete, symbols, keepalive


def create_builder(ctx: ContextHandle) -> BuilderHandle:
    out = ffi.new("AionBuilder**")
    status = lib.aion_builder_create(ctx.raw, out)
    raise_for_status(status, ctx, what="aion_builder_create")
    return BuilderHandle(out[0])


def destroy_builder(builder: BuilderHandle) -> None:
    lib.aion_builder_destroy(builder.raw)


def builder_value_shape(
    ctx: ContextHandle, builder: BuilderHandle, value: ValueId
) -> tuple[int, ...]:
    out_rank = ffi.new("size_t*")
    status = lib.aion_builder_value_rank(builder.raw, int(value), out_rank)
    raise_for_status(status, ctx, what="aion_builder_value_rank")
    rank = int(out_rank[0])
    dims = ffi.NULL if rank == 0 else ffi.new("size_t[]", rank)
    status = lib.aion_builder_value_shape(builder.raw, int(value), dims, rank)
    raise_for_status(status, ctx, what="aion_builder_value_shape")
    return tuple(int(dims[i]) for i in range(rank))


def builder_value_dtype(
    ctx: ContextHandle, builder: BuilderHandle, value: ValueId
) -> AionDType:
    out = ffi.new("AionDType*")
    status = lib.aion_builder_value_dtype(builder.raw, int(value), out)
    raise_for_status(status, ctx, what="aion_builder_value_dtype")
    return AionDType(int(out[0]))


def builder_value_dim_symbol(
    ctx: ContextHandle, builder: BuilderHandle, value: ValueId, axis: int
) -> str | None:
    """The dim symbol on `axis`, or None when that axis has a fixed size."""
    out_len = ffi.new("size_t*")
    buf = ffi.new("char[]", 128)
    status = lib.aion_builder_value_dim_symbol(
        builder.raw, int(value), int(axis), buf, 128, out_len
    )
    raise_for_status(status, ctx, what="aion_builder_value_dim_symbol")
    if int(out_len[0]) == 0:
        return None
    if int(out_len[0]) >= 128:
        buf = ffi.new("char[]", int(out_len[0]) + 1)
        status = lib.aion_builder_value_dim_symbol(
            builder.raw, int(value), int(axis), buf, int(out_len[0]) + 1, out_len
        )
        raise_for_status(status, ctx, what="aion_builder_value_dim_symbol")
    return ffi.string(buf).decode("utf-8")


def builder_symbol_size(
    ctx: ContextHandle, builder: BuilderHandle, name: str
) -> int:
    """The authoring placeholder size a dim symbol was declared with."""
    out = ffi.new("size_t*")
    status = lib.aion_builder_symbol_size(builder.raw, name.encode("utf-8"), out)
    raise_for_status(status, ctx, what="aion_builder_symbol_size")
    return int(out[0])


def builder_input(
    ctx: ContextHandle,
    builder: BuilderHandle,
    dtype: AionDType,
    shape: Sequence[int],
) -> ValueId:
    dims = tuple(int(dim) for dim in shape)
    c_shape = ffi.new("size_t[]", dims)
    out = ffi.new("AionValueId*")
    status = lib.aion_builder_input(
        builder.raw, int(dtype), len(dims), c_shape, out
    )
    raise_for_status(status, ctx, what="aion_builder_input")
    return ValueId(int(out[0]))


def builder_add_dim_symbol(
    ctx: ContextHandle,
    builder: BuilderHandle,
    value: ValueId,
    axis: int,
    name: str,
) -> None:
    status = lib.aion_builder_add_dim_symbol(
        builder.raw, int(value), int(axis), _string(name)
    )
    raise_for_status(status, ctx, what="aion_builder_add_dim_symbol")


def builder_param(
    ctx: ContextHandle,
    builder: BuilderHandle,
    tensor: TensorHandle,
) -> ValueId:
    out = ffi.new("AionValueId*")
    status = lib.aion_builder_param(builder.raw, tensor.raw, out)
    raise_for_status(status, ctx, what="aion_builder_param")
    return ValueId(int(out[0]))


def builder_name(
    ctx: ContextHandle, builder: BuilderHandle, value: ValueId, name: str
) -> None:
    status = lib.aion_builder_name(builder.raw, int(value), _string(name))
    raise_for_status(status, ctx, what="aion_builder_name")


def builder_param_named(
    ctx: ContextHandle, builder: BuilderHandle, tensor: TensorHandle, name: str
) -> ValueId:
    out = ffi.new("AionValueId*")
    status = lib.aion_builder_param_named(builder.raw, tensor.raw, _string(name), out)
    raise_for_status(status, ctx, what="aion_builder_param_named")
    return ValueId(int(out[0]))


def _read_str(fn: Callable[..., int], *args: Any) -> str:
    """Two-pass buf/cap/out_len read, the ABI's convention for returned strings."""
    out_len = ffi.new("size_t*")
    status = fn(*args, ffi.NULL, 0, out_len)
    if status != 0:
        return ""
    size = int(out_len[0])
    if size == 0:
        return ""
    buf = ffi.new("char[]", size + 1)
    status = fn(*args, buf, size + 1, out_len)
    if status != 0:
        return ""
    return ffi.string(buf, int(out_len[0])).decode("utf-8")


def builder_begin_scope(ctx: ContextHandle, builder: BuilderHandle, name: str) -> int:
    out = ffi.new("size_t*")
    status = lib.aion_builder_begin_scope(builder.raw, _string(name), out)
    raise_for_status(status, ctx, what="aion_builder_begin_scope")
    return int(out[0])


def builder_begin_auto_scope(
    ctx: ContextHandle, builder: BuilderHandle, base: str
) -> tuple[int, str]:
    """Returns `(depth, resolved segment)` — the segment is `{base}#{n}`."""
    out_depth = ffi.new("size_t*")
    out_len = ffi.new("size_t*")
    status = lib.aion_builder_begin_auto_scope(
        builder.raw, _string(base), out_depth, ffi.NULL, 0, out_len
    )
    raise_for_status(status, ctx, what="aion_builder_begin_auto_scope")
    # The scope is already open; read its name back from the builder's path.
    size = int(out_len[0])
    depth = int(out_depth[0])
    if size == 0:
        return depth, base
    path = builder_scope_path(builder)
    return depth, path.rsplit("/", 1)[-1] if path else base


def builder_end_scope(ctx: ContextHandle, builder: BuilderHandle, depth: int) -> None:
    status = lib.aion_builder_end_scope(builder.raw, int(depth))
    raise_for_status(status, ctx, what="aion_builder_end_scope")


def builder_scope_path(builder: BuilderHandle) -> str:
    return _read_str(lib.aion_builder_scope_path, builder.raw)


def builder_constant(ctx: ContextHandle, builder: BuilderHandle, value: float) -> ValueId:
    out = ffi.new("AionValueId*")
    status = lib.aion_builder_constant(builder.raw, float(value), out)
    raise_for_status(status, ctx, what="aion_builder_constant")
    return ValueId(int(out[0]))


def builder_filled_vec(
    ctx: ContextHandle, builder: BuilderHandle, dim: int, fill: float
) -> ValueId:
    out = ffi.new("AionValueId*")
    status = lib.aion_builder_filled_vec(builder.raw, int(dim), float(fill), out)
    raise_for_status(status, ctx, what="aion_builder_filled_vec")
    return ValueId(int(out[0]))


def builder_param_kind(builder: BuilderHandle, value: ValueId) -> int:
    """0 = not a parameter, 1 = user weight, 2 = synthesized constant."""
    out = ffi.new("uint32_t*")
    if lib.aion_builder_param_kind(builder.raw, int(value), out) != 0:
        return 0
    return int(out[0])


def builder_value_name(builder: BuilderHandle, value: ValueId) -> str:
    """The debug name attached to a value, or "" when it has none."""
    return _read_str(lib.aion_builder_value_name, builder.raw, int(value))


def builder_has_param_named(builder: BuilderHandle, name: str) -> bool:
    out = ffi.new("uint8_t*")
    if lib.aion_builder_has_param_named(builder.raw, _string(name), out) != 0:
        return False
    return int(out[0]) != 0


def emit_op(
    ctx: ContextHandle,
    builder: BuilderHandle,
    op: AionOp,
    inputs: Sequence[ValueId],
    attrs: OpAttrs = None,
) -> ValueId:
    spec = ffi.new("AionOpSpec*")
    spec.op = int(op)
    ids = ffi.new("AionValueId[]", [int(value) for value in inputs])
    spec.inputs = ids
    spec.inputs_len = len(inputs)
    keepalive: list[object] = [ids]

    if isinstance(attrs, MatmulAttrs):
        spec.attr.matmul.alpha = float(attrs.alpha)
        spec.attr.matmul.beta = float(attrs.beta)
    elif isinstance(attrs, ElemwiseAttrs):
        spec.attr.elemwise.op = int(attrs.op)
    elif isinstance(attrs, UnaryAttrs):
        spec.attr.unary.op = int(attrs.op)
    elif isinstance(attrs, AxisAttrs):
        member = {
            AionOp.AION_OP_SOFTMAX: spec.attr.softmax,
            AionOp.AION_OP_CONCAT: spec.attr.concat,
            AionOp.AION_OP_ARGMAX: spec.attr.argmax,
            AionOp.AION_OP_UNSQUEEZE: spec.attr.unsqueeze,
            AionOp.AION_OP_DIM: spec.attr.argmax,
            AionOp.AION_OP_IOTA: spec.attr.argmax,
        }.get(op)
        if member is None:
            raise TypeError(f"AxisAttrs are not valid for {op.name}")
        member.axis = int(attrs.axis)
    elif isinstance(attrs, NormAttrs):
        shape = ffi.new("size_t[]", attrs.normalized_shape)
        keepalive.append(shape)
        spec.attr.norm.eps = float(attrs.eps)
        spec.attr.norm.normalized_shape = shape
        spec.attr.norm.normalized_shape_len = len(attrs.normalized_shape)
    elif isinstance(attrs, Rope1DAttrs):
        spec.attr.rope1d.base_frequency = float(attrs.base_frequency)
        spec.attr.rope1d.scale_factor = float(attrs.scale_factor)
        spec.attr.rope1d.rope_proportion = float(attrs.rope_proportion)
    elif isinstance(attrs, ReshapeAttrs):
        shape, symbols, keep = _view_dims(attrs.dims)
        keepalive.extend(keep)
        spec.attr.reshape.shape = shape
        spec.attr.reshape.shape_len = len(attrs.dims.concrete)
        spec.attr.reshape.shape_symbols = symbols
    elif isinstance(attrs, CastAttrs):
        spec.attr.cast.to_dtype = int(attrs.dtype)
    elif isinstance(attrs, ReduceAttrs):
        spec.attr.reduce.op = int(attrs.op)
        spec.attr.reduce.axis = 0 if attrs.axis is None else int(attrs.axis)
        spec.attr.reduce.has_axis = 0 if attrs.axis is None else 1
    elif isinstance(attrs, Conv1DAttrs):
        spec.attr.conv1d.stride = int(attrs.stride)
        spec.attr.conv1d.dilation = int(attrs.dilation)
        spec.attr.conv1d.pad_left = int(attrs.pad_left)
        spec.attr.conv1d.pad_right = int(attrs.pad_right)
        spec.attr.conv1d.groups = int(attrs.groups)
        spec.attr.conv1d.pad_mode = int(attrs.pad_mode)
    elif isinstance(attrs, Conv2DAttrs):
        spec.attr.conv2d.stride_h = int(attrs.stride_h)
        spec.attr.conv2d.stride_w = int(attrs.stride_w)
        spec.attr.conv2d.dilation_h = int(attrs.dilation_h)
        spec.attr.conv2d.dilation_w = int(attrs.dilation_w)
        spec.attr.conv2d.pad_top = int(attrs.pad_top)
        spec.attr.conv2d.pad_bottom = int(attrs.pad_bottom)
        spec.attr.conv2d.pad_left = int(attrs.pad_left)
        spec.attr.conv2d.pad_right = int(attrs.pad_right)
        spec.attr.conv2d.groups = int(attrs.groups)
        spec.attr.conv2d.pad_mode = int(attrs.pad_mode)
    elif isinstance(attrs, StftAttrs):
        spec.attr.stft.n_fft = int(attrs.n_fft)
        spec.attr.stft.hop_length = int(attrs.hop_length)
        spec.attr.stft.center = 1 if attrs.center else 0
    elif isinstance(attrs, OptionalAxisAttrs):
        spec.attr.squeeze.axis = 0 if attrs.axis is None else int(attrs.axis)
        spec.attr.squeeze.has_axis = 0 if attrs.axis is None else 1
    elif isinstance(attrs, SliceAttrs):
        starts = ffi.new("size_t[]", attrs.starts)
        lens, symbols, keep = _view_dims(attrs.lens)
        keepalive.extend([starts, *keep])
        spec.attr.slice.starts = starts
        spec.attr.slice.lens = lens
        spec.attr.slice.len = len(attrs.starts)
        spec.attr.slice.len_symbols = symbols
    elif isinstance(attrs, AttentionAttrs):
        if op is not AionOp.AION_OP_ATTENTION:
            raise TypeError(f"AttentionAttrs are not valid for {op.name}")
        member = spec.attr.attention
        member.scale = float(attrs.scale)
        _set_window(member.window, attrs.window)
        member.attn_logits_soft_cap = float(attrs.attn_logits_soft_cap)
        member.has_query_positions = 1 if attrs.has_query_positions else 0
        member.has_kv_lengths = 1 if attrs.has_kv_lengths else 0
    elif isinstance(attrs, GatherAttrs):
        spec.attr.gather.axis = int(attrs.axis)
        spec.attr.gather.batch_dims = int(attrs.batch_dims)
    elif isinstance(attrs, RelposMhaAttrs):
        spec.attr.relpos_mha.scale = float(attrs.scale)
        _set_window(spec.attr.relpos_mha.window, attrs.window)
        spec.attr.relpos_mha.relative_zero_index = int(attrs.relative_zero_index)
        spec.attr.relpos_mha.attn_logits_soft_cap = float(attrs.attn_logits_soft_cap)

    out = ffi.new("AionValueId*")
    status = lib.aion_builder_op(builder.raw, spec, out)
    raise_for_status(status, ctx, what="aion_builder_op")
    return ValueId(int(out[0]))


def builder_add_output_alias(
    ctx: ContextHandle,
    builder: BuilderHandle,
    input_value: ValueId,
    output_value: ValueId,
) -> None:
    status = lib.aion_builder_add_output_alias(
        builder.raw, int(input_value), int(output_value)
    )
    raise_for_status(status, ctx, what="aion_builder_add_output_alias")


def builder_add_input_role(
    ctx: ContextHandle,
    builder: BuilderHandle,
    value: ValueId,
    kind: int,
    *,
    axis: int | None,
    capacity_symbol: str | None,
    zero_init: bool,
    growable: bool,
    ring: bool,
) -> None:
    capacity = ffi.NULL if capacity_symbol is None else _string(capacity_symbol)
    status = lib.aion_builder_add_input_role(
        builder.raw,
        int(value),
        int(kind),
        0 if axis is None else int(axis),
        0 if axis is None else 1,
        capacity,
        1 if zero_init else 0,
        1 if growable else 0,
        1 if ring else 0,
    )
    raise_for_status(status, ctx, what="aion_builder_add_input_role")


def builder_begin_region(ctx: ContextHandle, builder: BuilderHandle) -> None:
    status = lib.aion_builder_begin_region(builder.raw)
    raise_for_status(status, ctx, what="aion_builder_begin_region")


def builder_end_region(
    ctx: ContextHandle, builder: BuilderHandle, outputs: Sequence[ValueId]
) -> RegionId:
    ids = ffi.new("AionValueId[]", [int(value) for value in outputs])
    out = ffi.new("AionRegionId*")
    status = lib.aion_builder_end_region(builder.raw, ids, len(outputs), out)
    raise_for_status(status, ctx, what="aion_builder_end_region")
    return RegionId(int(out[0]))


def builder_if(
    ctx: ContextHandle,
    builder: BuilderHandle,
    cond: ValueId,
    then_region: RegionId,
    else_region: RegionId,
) -> ValueId:
    out = ffi.new("AionValueId*")
    status = lib.aion_builder_if(
        builder.raw,
        int(cond),
        int(then_region),
        int(else_region),
        out,
    )
    raise_for_status(status, ctx, what="aion_builder_if")
    return ValueId(int(out[0]))


def builder_loop(
    ctx: ContextHandle,
    builder: BuilderHandle,
    carried: Sequence[ValueId],
    body_region: RegionId,
    trip: int,
    *,
    cond_carry: int | None,
    check_before: bool,
) -> list[ValueId]:
    ids = ffi.new("AionValueId[]", [int(value) for value in carried])
    outs = ffi.new("AionValueId[]", len(carried))
    status = lib.aion_builder_loop(
        builder.raw,
        ids,
        len(carried),
        int(body_region),
        int(trip),
        0 if cond_carry is None else int(cond_carry),
        0 if cond_carry is None else 1,
        1 if check_before else 0,
        outs,
    )
    raise_for_status(status, ctx, what="aion_builder_loop")
    return [ValueId(int(outs[i])) for i in range(len(carried))]


def builder_mark_output(
    ctx: ContextHandle, builder: BuilderHandle, value: ValueId, name: str
) -> None:
    status = lib.aion_builder_mark_output(
        builder.raw, int(value), _string(name)
    )
    raise_for_status(status, ctx, what="aion_builder_mark_output")


def builder_clear_outputs(ctx: ContextHandle, builder: BuilderHandle) -> None:
    """Forget every output marked so far (see `aion_builder_clear_outputs`)."""
    status = lib.aion_builder_clear_outputs(builder.raw)
    raise_for_status(status, ctx, what="aion_builder_clear_outputs")


def builder_add_metadata(
    ctx: ContextHandle, builder: BuilderHandle, key: str, value: str
) -> None:
    status = lib.aion_builder_add_metadata(
        builder.raw, _string(key), _string(value)
    )
    raise_for_status(status, ctx, what="aion_builder_add_metadata")


def compile_builder(
    ctx: ContextHandle,
    builder: BuilderHandle,
    device_kind: int,
    device_index: int,
) -> ModelHandle:
    out = ffi.new("AionLoadedModel**")
    status = lib.aion_builder_compile(
        builder.raw, int(device_kind), int(device_index), out
    )
    raise_for_status(status, ctx, what="aion_builder_compile")
    return ModelHandle(out[0])


def export_builder(
    ctx: ContextHandle, builder: BuilderHandle, path: str
) -> None:
    status = lib.aion_builder_export_path(builder.raw, _string(path))
    raise_for_status(status, ctx, what="aion_builder_export_path")


__all__ = [
    "AttentionAttrs",
    "AttentionWindow",
    "AxisAttrs",
    "CastAttrs",
    "Conv1DAttrs",
    "Conv2DAttrs",
    "ElemwiseAttrs",
    "GatherAttrs",
    "MatmulAttrs",
    "NormAttrs",
    "OpAttrs",
    "OptionalAxisAttrs",
    "ReduceAttrs",
    "RegionId",
    "RelposMhaAttrs",
    "ReshapeAttrs",
    "Rope1DAttrs",
    "SliceAttrs",
    "StftAttrs",
    "UnaryAttrs",
    "ValueId",
    "ViewDims",
    "builder_add_dim_symbol",
    "builder_add_input_role",
    "builder_add_metadata",
    "builder_add_output_alias",
    "builder_begin_region",
    "builder_end_region",
    "builder_if",
    "builder_input",
    "builder_loop",
    "builder_mark_output",
    "builder_name",
    "builder_param",
    "builder_value_dtype",
    "builder_value_shape",
    "compile_builder",
    "create_builder",
    "destroy_builder",
    "emit_op",
    "export_builder",
]
