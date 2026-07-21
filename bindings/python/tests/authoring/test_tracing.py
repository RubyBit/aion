# SPDX-License-Identifier: Apache-2.0
"""Tracing front-end: aion.compile / aion.export / aion.spec + nn.Module.

Author a model as an nn.Module (weights held as Parameters, no builder threaded),
trace it under the ambient builder, and check results against a numpy reference.
"""
from __future__ import annotations

import numpy as np
import pytest

import aion
import aion.nn as nn


@pytest.fixture
def ctx():
    c = aion.Context(thread_count=1)
    try:
        yield c
    finally:
        c.close()


def test_mlp_compile_matches_numpy(ctx):
    w1 = np.random.RandomState(0).randn(4, 8).astype(np.float32)
    w2 = np.random.RandomState(1).randn(8, 3).astype(np.float32)

    class MLP(nn.Module):
        def __init__(self, a, b):
            self.fc1 = nn.Linear(a)
            self.fc2 = nn.Linear(b)

        def forward(self, x):
            return self.fc2(self.fc1(x).relu())

    mlp = MLP(w1, w2)
    assert len(list(mlp.parameters())) == 2

    model = aion.compile(mlp, aion.spec((None, 4)), ctx=ctx)
    x = np.random.RandomState(2).randn(5, 4).astype(np.float32)
    got = model.run({"x": x})["output0"]
    ref = np.maximum(x @ w1, 0.0) @ w2
    assert np.allclose(got, ref, atol=1e-4)


def test_input_named_from_forward_param(ctx):
    w = np.eye(4, dtype=np.float32)

    class Net(nn.Module):
        def __init__(self):
            self.fc = nn.Linear(w)

        def forward(self, tokens):  # parameter name becomes the input name
            return self.fc(tokens)

    model = aion.compile(Net(), aion.spec((1, 4)), ctx=ctx)
    assert model.input_names() == ["tokens"]
    got = model.run({"tokens": np.ones((1, 4), np.float32)})["output0"]
    assert np.allclose(got, np.ones((1, 4)), atol=1e-6)


def test_spec_name_overrides_param_name(ctx):
    w = np.eye(4, dtype=np.float32)

    class Net(nn.Module):
        def __init__(self):
            self.fc = nn.Linear(w)

        def forward(self, x):
            return self.fc(x)

    model = aion.compile(Net(), aion.spec((1, 4), name="feats"), ctx=ctx)
    assert model.input_names() == ["feats"]


def test_multi_input_and_dict_outputs(ctx):
    class Net(nn.Module):
        def forward(self, a, b):
            return {"sum": a + b, "prod": a * b}

    model = aion.compile(Net(), aion.spec((2, 3)), aion.spec((2, 3)), ctx=ctx)
    assert model.input_names() == ["a", "b"]
    av = np.full((2, 3), 2.0, np.float32)
    bv = np.full((2, 3), 3.0, np.float32)
    out = model.run({"a": av, "b": bv})
    assert np.allclose(out["sum"], 5.0) and np.allclose(out["prod"], 6.0)


def test_dynamic_axis_serves_many_sizes(ctx):
    w = np.arange(12, dtype=np.float32).reshape(4, 3)

    class Net(nn.Module):
        def __init__(self):
            self.fc = nn.Linear(w)

        def forward(self, x):
            return self.fc(x)

    model = aion.compile(Net(), aion.spec((None, 4)), ctx=ctx)
    for m in (1, 5, 2):
        xv = ((np.arange(m * 4, dtype=np.float32) % 7) - 3).reshape(m, 4)
        got = model.run({"x": xv})["output0"]
        assert got.shape == (m, 3)
        assert np.allclose(got, xv @ w, atol=1e-4)


def test_recompile_same_module_rebinds(ctx):
    w = np.eye(4, dtype=np.float32) * 2.0

    class Net(nn.Module):
        def __init__(self):
            self.fc = nn.Linear(w)

        def forward(self, x):
            return self.fc(x)

    net = Net()
    m1 = aion.compile(net, aion.spec((1, 4)), ctx=ctx)
    m2 = aion.compile(net, aion.spec((1, 4)), ctx=ctx)
    xv = np.ones((1, 4), np.float32)
    assert np.allclose(m1.run({"x": xv})["output0"], xv * 2.0, atol=1e-6)
    assert np.allclose(m2.run({"x": xv})["output0"], xv * 2.0, atol=1e-6)


def test_export_reload(ctx, tmp_path):
    w = np.eye(4, dtype=np.float32) * 3.0

    class Net(nn.Module):
        def __init__(self):
            self.fc = nn.Linear(w)

        def forward(self, x):
            return self.fc(x)

    path = tmp_path / "traced.aion"
    aion.export(Net(), aion.spec((None, 4)), str(path), ctx=ctx)
    assert path.exists() and path.stat().st_size > 0

    model = aion.load_model(str(path), thread_count=1)
    xv = np.array([[1.0, 2.0, 3.0, 4.0]], np.float32)
    assert np.allclose(model.run({"x": xv})["output0"], xv * 3.0, atol=1e-6)


def test_nn_layer_outside_trace_raises():
    w = np.eye(4, dtype=np.float32)
    with aion.Context(thread_count=1) as ctx:
        x = aion.Builder(ctx).input((1, 4))
        with pytest.raises(RuntimeError, match="aion.compile"):
            nn.Linear(w)(x)


def test_export_requires_path(ctx):
    class Net(nn.Module):
        def forward(self, x):
            return x

    with pytest.raises(TypeError, match="path"):
        aion.export(Net(), aion.spec((1, 4)), ctx=ctx)


def test_compile_requires_a_spec(ctx):
    class Net(nn.Module):
        def forward(self):
            raise AssertionError("should not run")

    with pytest.raises(ValueError, match="spec"):
        aion.compile(Net(), ctx=ctx)
