# SPDX-License-Identifier: Apache-2.0
"""Load a synthetic `.aion` package and run it through the Zig runtime."""
from __future__ import annotations

import numpy as np

import aion


def test_load_model_introspection_and_run(tiny_model):
    with aion.Context(thread_count=1) as ctx:
        with aion.LoadedModel.load(ctx, str(tiny_model.path)) as m:
            in_names = m.input_names()
            out_names = m.output_names()
            assert len(in_names) == m.input_count()
            assert len(out_names) == m.output_count()
            assert in_names == ["x"]
            assert out_names == ["y"]
            # A python-authored package must run through the Zig runtime and
            # produce the reference result (authoring/format lockstep guard).
            out = m.run_numpy({"x": tiny_model.x})["y"]
            expected = tiny_model.x @ tiny_model.w
            assert out.shape == expected.shape
            np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_load_model_q8_weights_run(tiny_q8_model):
    # q8_0-quantized weights authored via the C-ABI Builder must load and execute;
    # quantization error bounds the comparison (this is the q8 layout canary).
    with aion.Context(thread_count=1) as ctx:
        with aion.LoadedModel.load(ctx, str(tiny_q8_model.path)) as m:
            out = m.run_numpy({"x": tiny_q8_model.x})["y"]
            expected = tiny_q8_model.x @ tiny_q8_model.w
            np.testing.assert_allclose(out, expected, rtol=5e-2, atol=5e-2)


def test_output_handles_are_snapshots_of_their_run(tiny_model):
    # An output handle names one run's value. It used to share a host buffer with
    # every other fetch of that output, so it silently changed to whatever the next
    # fetch copied in (Silero's cached `prob` handle read one value forever).
    w = tiny_model.w
    x1 = tiny_model.x
    x2 = (x1 * -3.0 + 1.0).astype(np.float32)
    with aion.Context(thread_count=1) as ctx:
        with aion.LoadedModel.load(ctx, str(tiny_model.path)) as m:
            m.run_numpy({"x": x1})
            first = m.output_tensor("y")
            m.run_numpy({"x": x2})  # fetches `y` again internally
            second = m.output_tensor("y")
            np.testing.assert_allclose(first.numpy(), x1 @ w, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(second.numpy(), x2 @ w, rtol=1e-6, atol=1e-6)
            first.close()
            second.close()
