# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import pytest

import numpy as np

import aion


def test_tensor_numpy_roundtrip_f32():
    with aion.Context(thread_count=1) as ctx:
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        with aion.Tensor.from_numpy(ctx, arr) as t:
            out = t.numpy()
            assert out.dtype == np.float32
            assert out.shape == (2, 3)
            assert np.allclose(out, arr)


def test_tensor_write_from_numpy_f32():
    with aion.Context(thread_count=1) as ctx:
        with aion.Tensor.empty(ctx, (2, 3), dtype=aion.AionDType.AION_DTYPE_F32) as t:
            arr = (np.arange(6, dtype=np.float32) * 2.0).reshape(2, 3)
            t.write_from_numpy(arr)
            out = t.numpy()
            assert np.allclose(out, arr)


def test_numpy_shape_mismatch_errors_are_not_swallowed():
    with aion.Context(thread_count=1) as ctx:
        arr = np.zeros((2, 3), dtype=np.float32)

        # Tensor.from_f32 explicitly checks array shape when given a NumPy ndarray.
        with pytest.raises(ValueError, match=r"expected values with shape"):
            _t = aion.Tensor.from_f32(ctx, (6,), arr)  # same numel, different shape

        # Tensor.write_f32 should also preserve the NumPy shape mismatch error.
        with aion.Tensor.empty(ctx, (6,), dtype=aion.AionDType.AION_DTYPE_F32) as t2:
            with pytest.raises(ValueError, match=r"shape mismatch"):
                t2.write_f32(arr)
