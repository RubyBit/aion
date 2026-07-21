# SPDX-License-Identifier: Apache-2.0
"""Tensor construction, scalar/list I/O, and in-place helpers."""
from __future__ import annotations

import pytest

import aion


def test_tensor_scalar_roundtrip_f32():
    with aion.Context(thread_count=1) as ctx:
        with aion.Tensor.scalar_f32(ctx, 42.5) as t:
            assert t.read_scalar_f32() == pytest.approx(42.5)
            assert t.read_f32() == pytest.approx([42.5])


def test_tensor_write_read_f32():
    with aion.Context(thread_count=1) as ctx:
        with aion.Tensor.empty(ctx, (2, 3), dtype=aion.AionDType.AION_DTYPE_F32) as t:
            vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            t.write_f32(vals)
            assert t.read_f32() == pytest.approx(vals)


def test_tensor_constructor_simple_list_api():
    with aion.Tensor([[1.0, 2.0], [3.0, 4.0]]) as t:
        assert t.shape == (2, 2)
        assert t.read_f32() == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_tensor_constructor_scalar_api():
    with aion.Tensor(3.5) as t:
        assert t.read_scalar_f32() == pytest.approx(3.5)


def test_tensor_zeros_helper_api():
    with aion.Tensor.zeros((2, 3)) as t:
        assert t.shape == (2, 3)
        assert t.read_f32() == pytest.approx([0.0] * 6)


def test_tensor_zero_inplace_api():
    with aion.Tensor([1.0, 2.0, 3.0]) as t:
        t.zero()
        assert t.read_f32() == pytest.approx([0.0, 0.0, 0.0])


def test_tensor_zero_and_fill_aliases_api():
    with aion.Tensor([1.0, 2.0, 3.0]) as t:
        t.fill(0.25)
        assert t.read_f32() == pytest.approx([0.25, 0.25, 0.25])
        t.zero()
        assert t.read_f32() == pytest.approx([0.0, 0.0, 0.0])
