# SPDX-License-Identifier: Apache-2.0
"""Tensor construction, host conversion, and in-place helpers."""
from __future__ import annotations

import importlib

import pytest

import aion


def test_tensor_item():
    with aion.Context(thread_count=1) as ctx:
        with aion.Tensor(42.5, ctx=ctx) as t:
            assert t.item() == pytest.approx(42.5)
            assert t.tolist() == pytest.approx([42.5])


def test_tensor_copy_from_and_tolist():
    with aion.Context(thread_count=1) as ctx:
        with aion.Tensor.empty(ctx, (2, 3), dtype=aion.float32) as t:
            values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            assert t.copy_from(values) is t
            assert t.tolist() == values


def test_tensor_constructor_simple_list_api():
    with aion.Tensor([[1.0, 2.0], [3.0, 4.0]]) as t:
        assert t.shape == (2, 2)
        assert t.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_tensor_constructor_scalar_api():
    with aion.Tensor(3.5) as t:
        assert t.item() == pytest.approx(3.5)


def test_tensor_item_rejects_multiple_elements():
    with aion.Tensor([1.0, 2.0]) as t:
        with pytest.raises(ValueError, match="requires one element"):
            t.item()


def test_tensor_zeros_helper_api():
    with aion.Tensor.zeros((2, 3)) as t:
        assert t.shape == (2, 3)
        assert t.tolist() == [[0.0] * 3, [0.0] * 3]


def test_tensor_zero_inplace_api():
    with aion.Tensor([1.0, 2.0, 3.0]) as t:
        t.zero()
        assert t.tolist() == [0.0, 0.0, 0.0]


def test_tensor_zero_and_fill_aliases_api():
    with aion.Tensor([1.0, 2.0, 3.0]) as t:
        t.fill(0.25)
        assert t.tolist() == [0.25, 0.25, 0.25]
        t.zero()
        assert t.tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("dtype", "value", "expected"),
    [
        (aion.float32, 1.25, [[1.25, 1.25, 1.25]] * 2),
        (aion.float16, 1.5, [[1.5, 1.5, 1.5]] * 2),
        (aion.int8, -7, [[-7, -7, -7]] * 2),
        (aion.int32, 42, [[42, 42, 42]] * 2),
    ],
)
def test_copy_from_scalar_broadcasts(dtype, value, expected):
    with aion.Context(thread_count=1) as ctx:
        with aion.Tensor.empty(ctx, (2, 3), dtype=dtype) as tensor:
            assert tensor.copy_from(value) is tensor
            assert tensor.tolist() == expected


@pytest.mark.parametrize(
    ("dtype", "value", "expected"),
    [
        (aion.float32, 1.25, [1.25, 1.25]),
        (aion.float16, 1.5, [1.5, 1.5]),
        (aion.int8, -7, [-7, -7]),
        (aion.int32, 42, [42, 42]),
    ],
)
def test_copy_from_scalar_broadcasts_without_numpy(
    monkeypatch, dtype, value, expected
):
    tensor_module = importlib.import_module("aion.tensor")
    monkeypatch.setattr(tensor_module, "_try_numpy", lambda: None)

    with aion.Context(thread_count=1) as ctx:
        with aion.Tensor.empty(ctx, (2,), dtype=dtype) as tensor:
            tensor.copy_from(value)
            assert tensor.tolist() == expected


def test_copy_from_non_scalar_requires_exact_shape():
    with aion.Context(thread_count=1) as ctx:
        with aion.Tensor.empty(ctx, (2, 2)) as tensor:
            with pytest.raises(ValueError, match="shape mismatch"):
                tensor.copy_from([1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize("dtype", [aion.float16, aion.int8])
def test_non_numpy_constructor_supports_all_scalar_dtypes(monkeypatch, dtype):
    tensor_module = importlib.import_module("aion.tensor")
    monkeypatch.setattr(tensor_module, "_try_numpy", lambda: None)

    with aion.Tensor([1, 2], dtype=dtype) as tensor:
        assert tensor.tolist() == [1, 2]


def test_dtype_strings_and_raw_ordinals_are_rejected():
    with pytest.raises(TypeError, match="dtype strings are not supported"):
        aion.normalize_dtype("f16")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="raw dtype ordinals are not supported"):
        aion.normalize_dtype(1)  # type: ignore[arg-type]


def test_tensor_repr_uses_friendly_dtype_name():
    with aion.Tensor([1.0], dtype=aion.float16) as tensor:
        assert "dtype=float16" in repr(tensor)
