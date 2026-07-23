# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for the PyTorch-like authoring API.

Author a model in Python via `Builder`/`aion.nn`, then compile it in-process or
export it to `.aion` and reload — asserting results against a numpy reference.
"""
from __future__ import annotations

import numpy as np
import pytest

import aion
from aion import Builder, nn


@pytest.fixture
def ctx():
    c = aion.Context(thread_count=1)
    try:
        yield c
    finally:
        c.close()


def test_linear_relu_compile_matches_numpy(ctx):
    w = (np.arange(12, dtype=np.float32).reshape(4, 3) * 0.5) - 2.0

    class Net(nn.Module):
        def __init__(self):
            self.fc = nn.Linear(w)

        def forward(self, x):
            return self.fc(x).relu()

    model = aion.compile(Net(), aion.spec((1, 4)), ctx=ctx)
    xv = np.array([[2.0, -1.0, 0.5, 3.0]], dtype=np.float32)
    got = model.run({"x": xv})["output0"]
    ref = np.maximum(xv @ w, 0.0)
    assert np.allclose(got, ref, atol=1e-5)


def test_operator_overloading(ctx):
    b = Builder(ctx)
    x = b.input((1, 3)).rename("x")
    w = b.param(np.eye(3, dtype=np.float32) * 2.0)
    bias = b.param(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    y = (x @ w).rename("y")  # bias add via broadcast handled by nn; test @ here
    model = b.compile({"y": y})

    xv = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    got = model.run_numpy({"x": xv})["y"]
    assert np.allclose(got, xv @ (np.eye(3, dtype=np.float32) * 2.0), atol=1e-6)
    _ = bias  # param created & bound; not used in this graph


def test_q8_linear_within_tolerance(ctx):
    K, N = 64, 8
    w = ((np.arange(K * N, dtype=np.float32).reshape(K, N) % 7) - 3) * 0.05

    class Net(nn.Module):
        def __init__(self):
            self.fc = nn.Linear(w, dtype="q8_0")

        def forward(self, x):
            return self.fc(x)

    model = aion.compile(Net(), aion.spec((1, K)), ctx=ctx)
    xv = ((np.arange(K, dtype=np.float32) % 5) - 2) * 0.1
    got = model.run({"x": xv.reshape(1, K)})["output0"]
    ref = xv.reshape(1, K) @ w
    relerr = np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-9)
    assert relerr < 0.02  # 8-bit block quantization


def test_export_reload_roundtrip(ctx, tmp_path):
    w = np.eye(4, dtype=np.float32) * 3.0

    class Net(nn.Module):
        def __init__(self):
            self.fc = nn.Linear(w)

        def forward(self, x):
            return self.fc(x)

    path = tmp_path / "authored.aion"
    aion.export(Net(), aion.spec((1, 4)), str(path), ctx=ctx)
    assert path.exists() and path.stat().st_size > 0

    model = aion.load_model(str(path), thread_count=1)
    xv = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    got = model.run({"x": xv})["output0"]
    assert np.allclose(got, xv * 3.0, atol=1e-6)


def test_symbolic_shape_one_model_many_sizes(ctx):
    # A free row axis: compile ONCE, run for any number of rows. This is the
    # prefill (many tokens) -> decode (one token) pattern on one model. The
    # declared 8 is the authoring placeholder; the model still serves any size.
    b = Builder(ctx)
    x = b.input((8, 4), dynamic=(0,)).rename("x")
    w = np.arange(12, dtype=np.float32).reshape(4, 3)
    y = (x @ b.param(w)).rename("y")
    model = b.compile({"y": y})

    for m in (2, 5, 1):
        xv = ((np.arange(m * 4, dtype=np.float32) % 7) - 3).reshape(m, 4)
        got = model.run_numpy({"x": xv})["y"]
        assert got.shape == (m, 3)
        assert np.allclose(got, xv @ w, atol=1e-4)


def test_shared_symbol_name_constrains_axes(ctx):
    # Reusing a symbol name ties two inputs' axes to the same runtime size.
    b = Builder(ctx)
    a = b.input((4, 3), dynamic={0: "n"}).rename("a")
    c = b.input((4, 3), dynamic={0: "n"}).rename("c")
    y = (a + c).rename("y")
    model = b.compile({"y": y})

    av = np.ones((4, 3), dtype=np.float32)
    cv = np.full((4, 3), 2.0, dtype=np.float32)
    got = model.run_numpy({"a": av, "c": cv})["y"]
    assert np.allclose(got, av + cv)


def test_value_shape_introspection(ctx):
    # Shapes are queryable while authoring (eager per-op inference), and the
    # dynamic-axis placeholder propagates deterministically to derived values.
    b = Builder(ctx)
    x = b.input((8, 4), dynamic=(0,)).rename("x")
    assert x.shape == (8, 4)
    assert x.ndim == 2
    assert x.dtype == aion.AionDType.AION_DTYPE_F32

    w = np.arange(12, dtype=np.float32).reshape(4, 3)
    y = x @ b.param(w)
    assert y.shape == (8, 3)  # placeholder (8) carried through matmul


def test_shape_error_raised_at_offending_op(ctx):
    # A miswired op fails at the op call, not deferred to compile.
    b = Builder(ctx)
    x = b.input((2, 4))
    w = np.arange(9, dtype=np.float32).reshape(3, 3)  # K=3 != x's K=4
    with pytest.raises(aion.AionError):
        _ = x @ b.param(w)


def test_if_control_flow(ctx):
    b = Builder(ctx)
    cond = b.input((1,), dtype="i32").rename("cond")
    then_v = b.input((1,)).rename("then_v")
    else_v = b.input((1,)).rename("else_v")

    b.begin_region()
    then_region = b.end_region([then_v])
    b.begin_region()
    else_region = b.end_region([else_v])
    out = b.if_(cond, then_region, else_region).rename("out")

    model = b.compile({"out": out})
    model.bind_input("cond", aion.Tensor([1], ctx=ctx, dtype=aion.AionDType.AION_DTYPE_I32))
    model.bind_input("then_v", aion.Tensor([42.0], ctx=ctx))
    model.bind_input("else_v", aion.Tensor([7.0], ctx=ctx))
    model.run()
    assert model.output_tensor("out").item() == 42.0


def test_loop_control_flow(ctx):
    b = Builder(ctx)
    carried = b.input((1,)).rename("carried")
    inc = b.input((1,)).rename("inc")

    b.begin_region()
    nxt = carried + inc  # operator overloading inside the loop body
    body = b.end_region([nxt])
    out = b.loop(carried, body, 4).rename("out")

    model = b.compile({"out": out})
    model.bind_input("carried", aion.Tensor([1.0], ctx=ctx))
    model.bind_input("inc", aion.Tensor([2.0], ctx=ctx))
    model.run()
    # 1 + 2*4 = 9
    assert abs(model.output_tensor("out").item() - 9.0) < 1e-6


def test_rmsnorm_forward(ctx):
    D = 8
    gamma = np.ones(D, dtype=np.float32)

    class Net(nn.Module):
        def __init__(self):
            self.norm = nn.RMSNorm(gamma, eps=1e-6, normalized_shape=[D])

        def forward(self, x):
            return self.norm(x)

    model = aion.compile(Net(), aion.spec((1, D)), ctx=ctx)
    xv = np.linspace(-2.0, 2.0, D, dtype=np.float32).reshape(1, D)
    got = model.run({"x": xv})["output0"]
    rms = np.sqrt(np.mean(xv**2) + 1e-6)
    ref = xv / rms
    assert np.allclose(got, ref, atol=1e-4)
