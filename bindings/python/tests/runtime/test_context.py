# SPDX-License-Identifier: Apache-2.0
"""Context lifecycle + the process-wide default context."""
from __future__ import annotations

import importlib

import pytest

import aion
import aion.context as context_module


def test_context_create_destroy():
    with aion.Context(thread_count=1):
        pass


def test_gpu_context_adds_missing_runtime_hint(monkeypatch):
    native_error = aion.AionError(
        aion.AionStatus.AION_UNSUPPORTED,
        "aion_context_create: AION_UNSUPPORTED",
    )

    def fail_create(_thread_count, _gpus):
        raise native_error

    monkeypatch.setattr(context_module, "create_context", fail_create)
    monkeypatch.setattr(
        context_module,
        "missing_wgpu_runtime_hint",
        lambda: "The optional wgpu-native runtime is not installed. Run `uv sync`.",
    )

    with pytest.raises(
        aion.AionError,
        match=r"wgpu-native runtime is not installed.*uv sync",
    ):
        aion.Context.gpu(thread_count=1)


def test_context_close_auto_closes_live_children():
    ctx = aion.Context(thread_count=1)
    _t = aion.Tensor.empty(ctx, (2, 2), dtype=aion.float32)
    # Should not raise even though child tensor is still live.
    ctx.close()


def test_children_closed_after_their_context_leave_it_alone(tiny_model, monkeypatch):
    # At interpreter exit the garbage collector clears the context's weak set of
    # children before finalizers run, so the context can close first. Its teardown
    # freed everything a child points at, so a child closed after it must not call
    # into the library at all.
    ctx = aion.Context(thread_count=1)
    m = aion.LoadedModel.load(ctx, str(tiny_model.path))
    x = aion.Tensor(tiny_model.x, ctx=ctx)
    m.bind_input("x", x)
    m.run()
    b = aion.Builder(ctx)
    ctx._children.clear()
    ctx.close()

    calls = []
    for module, name in (("tensor", "destroy_tensor"), ("model", "destroy_model"), ("builder", "destroy_builder")):
        # By module path: `aion.tensor` is also the name of a function.
        monkeypatch.setattr(importlib.import_module(f"aion.{module}"), name, lambda handle, name=name: calls.append(name))
    for child in (x, m, b):
        child.close()
        child.__del__()
    assert calls == []


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
            assert t.tolist() == pytest.approx([1.0, 2.0, 3.0])
    finally:
        aion.reset_default_context()


def test_default_context_invalid_thread_env_raises(monkeypatch):
    aion.reset_default_context()
    monkeypatch.setenv("AION_DEFAULT_THREAD_COUNT", "nope")
    with pytest.raises(ValueError, match=r"must be an integer"):
        _t = aion.Tensor([1.0])
    aion.reset_default_context()
