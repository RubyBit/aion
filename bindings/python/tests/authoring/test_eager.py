# SPDX-License-Identifier: Apache-2.0
"""Unified lazy `Tensor`: eager compute, realize, and Tensor-first authoring.

Operators on tensors accumulate a lazy graph; `.numpy()`/`.realize()` compiles
and runs it. A graph rooted entirely in concrete data needs no explicit
`Builder` — that is the numpy/PyTorch-seamless path. Free-input graphs are
compiled into runnable models with `aion.compile(outputs, inputs=[...])`.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

import aion


@pytest.fixture
def ctx():
    c = aion.Context(thread_count=1)
    try:
        yield c
    finally:
        c.close()


# --- layering -------------------------------------------------------------
def test_value_and_tensor_are_distinct_layers():
    # Tensor is the high-level lazy/concrete value; Value is the low-level
    # builder handle returned by the raw Builder escape hatch.
    assert aion.Value is not aion.Tensor
    b = aion.Builder(ctx=aion.Context(thread_count=1))
    x = b.input((1, 2))
    assert isinstance(x, aion.Value)
    assert not isinstance(x, aion.Tensor)


def test_lazy_shape_before_realize(ctx):
    a = aion.tensor(np.zeros((2, 4), np.float32), ctx=ctx)
    w = aion.tensor(np.zeros((4, 3), np.float32), ctx=ctx)
    y = (a @ w).relu()
    assert y.shape == (2, 3)  # inferred via the core, no realize
    assert y._lazy  # still lazy after a shape query


def test_dtype_inference(ctx):
    ti = aion.tensor(np.array([1, 2, 3], np.int32), ctx=ctx)
    tf = aion.tensor([[1.0, 2.0]], ctx=ctx)
    ts = aion.tensor([1, 2, 3], dtype="i32", ctx=ctx)
    assert ti.dtype == aion.AionDType.AION_DTYPE_I32
    assert tf.dtype == aion.AionDType.AION_DTYPE_F32
    assert ts.dtype == aion.AionDType.AION_DTYPE_I32


# --- eager compute on concrete data --------------------------------------
def test_eager_matmul_relu_matches_numpy(ctx):
    a_np = np.random.RandomState(0).randn(2, 4).astype(np.float32)
    w_np = np.random.RandomState(1).randn(4, 3).astype(np.float32)
    a = aion.tensor(a_np, ctx=ctx)
    w = aion.tensor(w_np, ctx=ctx)

    y = (a @ w).relu()
    assert y._lazy  # lazy until materialized
    got = y.numpy()
    assert not y._lazy  # realized in place

    ref = np.maximum(a_np @ w_np, 0.0)
    assert np.allclose(got, ref, atol=1e-5)


def test_eager_scalar_ops(ctx):
    b_np = np.array([[1.0, 2.0, 3.0]], np.float32)
    b = aion.tensor(b_np, ctx=ctx)

    assert np.allclose(((b + 1.0) * 2.0).numpy(), (b_np + 1.0) * 2.0)
    assert np.allclose((10.0 - aion.tensor(b_np, ctx=ctx)).numpy(), 10.0 - b_np)
    assert np.allclose((-aion.tensor(b_np, ctx=ctx)).numpy(), -b_np)


def test_eager_numpy_operand(ctx):
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
    a = aion.tensor(a_np, ctx=ctx)
    # a raw numpy array as the right operand is lifted as a param
    got = (a + np.ones((2, 2), np.float32)).numpy()
    assert np.allclose(got, a_np + 1.0)


def test_i32_scalar_with_i32_tensor(ctx):
    a = aion.tensor(np.array([1, 2, 3], np.int32), ctx=ctx)
    assert (a + 5).numpy().tolist() == [6, 7, 8]
    with pytest.raises(TypeError):
        (a + 0.5).numpy()  # non-integer scalar on an i32 tensor


# --- realize semantics ----------------------------------------------------
def test_realize_group_multi_output(ctx):
    a_np = np.array([[1.0, -2.0]], np.float32)
    a = aion.tensor(a_np, ctx=ctx)
    g = a + a
    p1 = g.relu()
    p2 = g * g
    aion.realize([p1, p2])
    assert np.allclose(p1.numpy(), np.maximum(2 * a_np, 0.0))
    assert np.allclose(p2.numpy(), (2 * a_np) ** 2)


def test_shared_intermediate_independent_realize(ctx):
    # Two branches off a shared lazy intermediate realize independently — the
    # immutable DAG re-lowers each time, so there is no single-shot limitation.
    a = aion.tensor(np.array([[1.0, -2.0]], np.float32), ctx=ctx)
    g = a + a
    p1 = g.relu()
    p2 = g * g
    assert np.allclose(p1.numpy(), np.maximum(2 * a.numpy(), 0.0))
    assert np.allclose(p2.numpy(), (2 * a.numpy()) ** 2)


def test_combine_independent_chains(ctx):
    # Two separately-built lazy chains combine freely (no shared builder).
    a = aion.tensor(np.array([[1.0, -2.0, 3.0, 4.0]], np.float32), ctx=ctx)
    wa = aion.tensor(np.eye(4, 3, dtype=np.float32), ctx=ctx)
    b = aion.tensor(np.array([[2.0, 0.0, 1.0, -1.0]], np.float32), ctx=ctx)
    wb = aion.tensor(np.eye(4, 3, dtype=np.float32), ctx=ctx)
    z = (a @ wa).relu() + (b @ wb).relu()
    ref = np.maximum(a.numpy() @ wa.numpy(), 0.0) + np.maximum(b.numpy() @ wb.numpy(), 0.0)
    assert np.allclose(z.numpy(), ref, atol=1e-5)


def test_realize_free_input_raises(ctx):
    x = aion.tensor(shape=(1, 4), name="x", ctx=ctx)
    y = x.relu()
    with pytest.raises(RuntimeError):
        y.numpy()


def test_cross_context_raises():
    c1 = aion.Context(thread_count=1)
    c2 = aion.Context(thread_count=1)
    try:
        a = aion.tensor(np.ones((1, 2), np.float32), ctx=c1)
        b = aion.tensor(np.ones((1, 2), np.float32), ctx=c2)
        with pytest.raises(ValueError):
            (a + b).numpy()
    finally:
        c1.close()
        c2.close()


def test_realize_is_idempotent(ctx):
    a = aion.tensor(np.array([[1.0, 2.0]], np.float32), ctx=ctx)
    y = a + a
    y.realize()
    first = y.numpy().copy()
    y.realize()  # no-op on a concrete tensor
    assert np.allclose(y.numpy(), first)


# --- Tensor-first authoring ----------------------------------------------
def test_tensor_first_compile_matches_numpy(ctx):
    w_np = np.random.RandomState(2).randn(4, 3).astype(np.float32)
    x = aion.tensor(shape=(1, 4), name="x", ctx=ctx)
    w = aion.tensor(w_np, ctx=ctx)
    out = (x @ w).relu()
    model = aion.compile({"y": out}, inputs=[x], ctx=ctx)

    assert model.input_names() == ["x"]
    assert model.output_names() == ["y"]
    xv = np.array([[1.0, -2.0, 3.0, -4.0]], np.float32)
    got = model.run_numpy({"x": xv})["y"]
    assert np.allclose(got, np.maximum(xv @ w_np, 0.0), atol=1e-5)


def test_tensor_first_export_reload(ctx):
    w_np = np.eye(4, 3).astype(np.float32)
    x = aion.tensor(shape=(1, 4), name="x", ctx=ctx)
    w = aion.tensor(w_np, ctx=ctx)
    out = x @ w
    path = os.path.join(tempfile.gettempdir(), "aion_eager_export.aion")
    aion.export({"y": out}, path, inputs=[x])

    model = aion.load_model(path)
    xv = np.array([[1.0, 2.0, 3.0, 4.0]], np.float32)
    got = model.run_numpy({"x": xv})["y"]
    assert np.allclose(got, xv @ w_np, atol=1e-5)


def test_graph_tensor_write_rejected(ctx):
    x = aion.tensor(shape=(1, 4), name="x", ctx=ctx)
    with pytest.raises(TypeError):
        x.copy_from([1.0, 2.0, 3.0, 4.0])
