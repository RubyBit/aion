# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import numpy as np

import aion


def test_tensor_numpy_roundtrip_f32():
    with aion.Context(thread_count=1) as ctx:
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        with aion.Tensor(arr, ctx=ctx) as t:
            out = t.numpy()
            assert out.dtype == np.float32
            assert out.shape == (2, 3)
            assert np.allclose(out, arr)


def test_tensor_copy_from_numpy_f32():
    with aion.Context(thread_count=1) as ctx:
        with aion.Tensor.empty(ctx, (2, 3), dtype=aion.float32) as t:
            arr = (np.arange(6, dtype=np.float32) * 2.0).reshape(2, 3)
            t.copy_from(arr)
            out = t.numpy()
            assert np.allclose(out, arr)


@pytest.mark.parametrize(
    ("dtype", "scalar"),
    [
        (aion.float32, np.float32(1.25)),
        (aion.float16, np.float16(1.5)),
        (aion.int8, np.int8(-7)),
        (aion.int32, np.int32(42)),
    ],
)
def test_tensor_copy_from_numpy_scalar_broadcasts(dtype, scalar):
    with aion.Context(thread_count=1) as ctx:
        with aion.Tensor.empty(ctx, (2, 3), dtype=dtype) as tensor:
            tensor.copy_from(scalar)
            assert np.all(tensor.numpy() == scalar)


@pytest.mark.parametrize(
    ("numpy_dtype", "expected"),
    [
        (np.float32, aion.float32),
        (np.dtype(np.float16), aion.float16),
        (np.int8, aion.int8),
        (np.dtype(np.int32), aion.int32),
    ],
)
def test_numpy_dtype_inputs_are_supported(numpy_dtype, expected):
    assert aion.normalize_dtype(numpy_dtype) == expected


def test_tensor_tolist_decodes_f16_values():
    values = np.array([[1.5, -2.25]], dtype=np.float16)
    with aion.Tensor(values) as tensor:
        assert tensor.tolist() == [[1.5, -2.25]]


def test_integer_data_infers_i32():
    with aion.Tensor([1, 2, 3]) as tensor:
        assert tensor.dtype == aion.int32
        assert tensor.tolist() == [1, 2, 3]


def test_numpy_shape_mismatch_is_not_swallowed():
    with aion.Context(thread_count=1) as ctx:
        arr = np.zeros((2, 3), dtype=np.float32)

        with aion.Tensor.empty(ctx, (6,), dtype=aion.float32) as t2:
            with pytest.raises(ValueError, match=r"shape mismatch"):
                t2.copy_from(arr)
