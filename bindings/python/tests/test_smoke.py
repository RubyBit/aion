# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

import aion


def test_context_create_destroy():
    with aion.Context(thread_count=1):
        pass


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
            out = t.read_f32()
            assert out == pytest.approx(vals)


def test_load_model_introspection_and_run(tiny_model):
    with aion.Context(thread_count=1) as ctx:
        with aion.LoadedModel.load(ctx, str(tiny_model.path)) as m:
            in_names = m.input_names()
            out_names = m.output_names()
            assert len(in_names) == m.input_count()
            assert len(out_names) == m.output_count()
            assert in_names == ["x"]
            assert out_names == ["y"]
            # Execute: a python-written package must run through the Zig runtime
            # and produce the reference result (format-lockstep guard).
            out = m.run_numpy({"x": tiny_model.x})["y"]
            expected = tiny_model.x @ tiny_model.w
            assert out.shape == expected.shape
            np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_load_model_q8_weights_run(tiny_q8_model):
    # q8_0-quantized weights written by the python packer must load and execute;
    # quantization error bounds the comparison.
    with aion.Context(thread_count=1) as ctx:
        with aion.LoadedModel.load(ctx, str(tiny_q8_model.path)) as m:
            out = m.run_numpy({"x": tiny_q8_model.x})["y"]
            expected = tiny_q8_model.x @ tiny_q8_model.w
            np.testing.assert_allclose(out, expected, rtol=5e-2, atol=5e-2)


def test_context_close_auto_closes_live_children():
    ctx = aion.Context(thread_count=1)
    _t = aion.Tensor.empty(ctx, (2, 2), dtype=aion.AionDType.AION_DTYPE_F32)
    # Should not raise even though child tensor is still live.
    ctx.close()


def test_tensor_constructor_simple_list_api():
    with aion.Tensor([[1.0, 2.0], [3.0, 4.0]]) as t:
        assert t.shape() == (2, 2)
        assert t.read_f32() == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_tensor_constructor_scalar_api():
    with aion.Tensor(3.5) as t:
        assert t.read_scalar_f32() == pytest.approx(3.5)


def test_tensor_zeros_helper_api():
    with aion.Tensor.zeros((2, 3)) as t:
        assert t.shape() == (2, 3)
        assert t.read_f32() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


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


def test_load_model_uses_default_context_when_thread_count_omitted(tiny_model_path):
    model_path = tiny_model_path
    aion.reset_default_context()
    try:
        with aion.load_model(str(model_path)) as m:
            with aion.Tensor([0.0]) as x:
                # Both should use the shared default context.
                assert m.context is x._ctx_owner
    finally:
        aion.reset_default_context()


def test_default_context_thread_count_from_env(monkeypatch):
    aion.reset_default_context()
    monkeypatch.setenv("AION_DEFAULT_THREAD_COUNT", "1")
    try:
        with aion.Tensor([1.0, 2.0, 3.0]) as t:
            assert t.shape() == (3,)
            assert t.read_f32() == pytest.approx([1.0, 2.0, 3.0])
    finally:
        aion.reset_default_context()


def test_default_context_invalid_thread_env_raises(monkeypatch):
    aion.reset_default_context()
    monkeypatch.setenv("AION_DEFAULT_THREAD_COUNT", "nope")
    with pytest.raises(ValueError, match=r"must be an integer"):
        _t = aion.Tensor([1.0])
    aion.reset_default_context()
