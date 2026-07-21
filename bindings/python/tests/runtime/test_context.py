# SPDX-License-Identifier: Apache-2.0
"""Context lifecycle + the process-wide default context."""
from __future__ import annotations

import pytest

import aion


def test_context_create_destroy():
    with aion.Context(thread_count=1):
        pass


def test_context_close_auto_closes_live_children():
    ctx = aion.Context(thread_count=1)
    _t = aion.Tensor.empty(ctx, (2, 2), dtype=aion.AionDType.AION_DTYPE_F32)
    # Should not raise even though child tensor is still live.
    ctx.close()


def test_load_model_uses_default_context_when_thread_count_omitted(tiny_model_path):
    aion.reset_default_context()
    try:
        with aion.load_model(str(tiny_model_path)) as m:
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
            assert t.shape == (3,)
            assert t.read_f32() == pytest.approx([1.0, 2.0, 3.0])
    finally:
        aion.reset_default_context()


def test_default_context_invalid_thread_env_raises(monkeypatch):
    aion.reset_default_context()
    monkeypatch.setenv("AION_DEFAULT_THREAD_COUNT", "nope")
    with pytest.raises(ValueError, match=r"must be an integer"):
        _t = aion.Tensor([1.0])
    aion.reset_default_context()
