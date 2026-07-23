"""Static contract tests; this module is checked by Pyright, not executed."""
from typing import assert_type

from aion import AionDType, Builder, Context, LoadedModel, Tensor, Value
from aion.tensor import TensorData
from aion._ffi import ContextHandle, ModelHandle, TensorHandle


ctx = Context(thread_count=1)
assert_type(ctx.ptr, ContextHandle)

tensor = Tensor.empty(ctx, (2, 3), dtype=AionDType.AION_DTYPE_F32)
assert_type(tensor.ptr, TensorHandle)
assert_type(tensor.shape, tuple[int, ...])
assert_type(tensor.dtype, AionDType)
assert_type(tensor.tolist(), TensorData)
assert_type(tensor.item(), int | float)
assert_type(tensor.copy_from([[1, 2, 3], [4, 5, 6]]), Tensor)
assert_type(tensor.to("cpu"), Tensor)

builder = Builder(ctx)
value = builder.input((2, 3))
assert_type(value, Value)
assert_type(value.shape, tuple[int, ...])
assert_type(value.dtype, AionDType)

model = builder.compile({"output": value})
assert_type(model, LoadedModel)
assert_type(model.ptr, ModelHandle)
assert_type(model.input_names(), list[str])
assert_type(model.output_names(), list[str])
