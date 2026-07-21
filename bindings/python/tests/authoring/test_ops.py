# SPDX-License-Identifier: Apache-2.0
"""Per-op parity for the Builder op wrappers added for converter migration.

Each test authors a tiny graph via the C-ABI Builder, compiles it, runs it, and
checks against a numpy reference. Ops needing large real-model shapes/oracles
(conv2d, stft/rfft, lstm_cell, mha/relpos/mha_cached, sequence_append,
scatter_row) are covered by the converter parity harnesses in Stage 4.
"""
from __future__ import annotations

import numpy as np
import pytest

import aion
from aion import Builder


@pytest.fixture
def ctx():
    c = aion.Context(thread_count=1)
    try:
        yield c
    finally:
        c.close()


def test_slice_concrete(ctx):
    with Builder(ctx) as b:
        x = b.input((4, 3)).rename("x")
        y = b.slice(x, (1, 0), (2, 3)).rename("y")
        m = b.compile({"y": y})
        xv = np.arange(12, dtype=np.float32).reshape(4, 3)
        assert np.allclose(m.run({"x": xv})["y"], xv[1:3, :])


def test_unsqueeze_squeeze_roundtrip(ctx):
    with Builder(ctx) as b:
        x = b.input((2, 4)).rename("x")
        y = b.squeeze(b.unsqueeze(x, 0), 0).rename("y")
        m = b.compile({"y": y})
        xv = np.arange(8, dtype=np.float32).reshape(2, 4)
        got = m.run({"x": xv})["y"]
        assert got.shape == (2, 4) and np.allclose(got, xv)


@pytest.mark.parametrize("op,fn", [
    ("eq", np.equal), ("ne", np.not_equal),
    ("lt", np.less), ("gt", np.greater),
    ("le", np.less_equal), ("ge", np.greater_equal),
])
def test_compare_i32(ctx, op, fn):
    with Builder(ctx) as b:
        a = b.input((2, 3), dtype="i32").rename("a")
        c = b.input((2, 3), dtype="i32").rename("c")
        y = b.compare(op, a, c).rename("y")
        m = b.compile({"y": y})
        av = np.array([[1, 5, 3], [4, 2, 6]], np.int32)
        cv = np.full((2, 3), 3, np.int32)
        got = m.run_numpy({"a": av, "c": cv})["y"]
        assert np.array_equal(got, fn(av, cv).astype(np.int32))


@pytest.mark.parametrize("method,pyfn", [
    ("broadcast_mul", lambda x, v: x * v),
    ("broadcast_sub", lambda x, v: x - v),
    ("broadcast_div", lambda x, v: x / v),
])
def test_broadcast_last_dim(ctx, method, pyfn):
    with Builder(ctx) as b:
        x = b.input((2, 3)).rename("x")
        v = b.input((3,)).rename("v")
        y = getattr(b, method)(x, v).rename("y")
        m = b.compile({"y": y})
        xv = (np.arange(6, dtype=np.float32) + 1).reshape(2, 3)
        vv = np.array([1.0, 2.0, 4.0], np.float32)
        assert np.allclose(m.run({"x": xv, "v": vv})["y"], pyfn(xv, vv), atol=1e-6)


def test_copy(ctx):
    with Builder(ctx) as b:
        x = b.input((2, 3)).rename("x")
        y = b.copy(x).rename("y")
        m = b.compile({"y": y})
        xv = np.arange(6, dtype=np.float32).reshape(2, 3)
        assert np.allclose(m.run({"x": xv})["y"], xv)


def test_conv1d(ctx):
    # Aion conv1d: x [..., L, C_in], w [K, C_in/groups, C_out], channels last.
    with Builder(ctx) as b:
        x = b.input((1, 5, 1)).rename("x")
        w = b.param(np.array([[[1.0]], [[0.0]], [[-1.0]]], dtype=np.float32))  # [K=3, Cin=1, Cout=1]
        y = b.conv1d(x, w).rename("y")
        m = b.compile({"y": y})
        xv = np.arange(5, dtype=np.float32).reshape(1, 5, 1)
        got = m.run({"x": xv})["y"]
        # cross-correlation with [1,0,-1]: out[l] = x[l] - x[l+2]
        ref = np.array([[[0.0 - 2.0], [1.0 - 3.0], [2.0 - 4.0]]], np.float32)
        assert got.shape == (1, 3, 1) and np.allclose(got, ref)
