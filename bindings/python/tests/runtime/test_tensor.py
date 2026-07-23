# SPDX-License-Identifier: Apache-2.0
"""Tensor construction, host conversion, and in-place helpers."""
from __future__ import annotations

import pytest

import aion


def test_tensor_item():
    with aion.Context(thread_count=1) as ctx:
        with aion.Tensor(42.5, ctx=ctx) as t:
            assert t.item() == pytest.approx(42.5)
            assert t.tolist() == pytest.approx([42.5])


def test_tensor_copy_from_and_tolist():
    with aion.Context(thread_count=1) as ctx:
        with aion.Tensor.empty(ctx, (2, 3), dtype=aion.AionDType.AION_DTYPE_F32) as t:
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
