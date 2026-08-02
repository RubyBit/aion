# SPDX-License-Identifier: Apache-2.0
"""Per-op parity for the Builder op wrappers added for converter migration.

Each test authors a tiny graph via the C-ABI Builder, compiles it, runs it, and
checks against a numpy reference. Ops needing large real-model shapes/oracles
(conv2d, stft/rfft, lstm_cell, attention/relpos, sequence_append,
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


def test_shape_iota_dim_and_batched_gather(ctx):
    with Builder(ctx) as b:
        data = b.input((2, 4, 3)).rename("data")
        indices = b.input((2, 2), dtype=aion.int32).rename("indices")
        tokens = b.input((2, 4), dtype=aion.int32).rename("tokens")
        outputs = {
            "gathered": b.gather(data, indices, axis=1, batch_dims=1),
            "positions": b.iota(tokens, axis=1),
            "seq_len": b.dim(tokens, axis=1),
        }
        model = b.compile(outputs)

        data_value = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
        indices_value = np.array([[3, 1], [0, 2]], dtype=np.int32)
        got = model.run_numpy({
            "data": data_value,
            "indices": indices_value,
            "tokens": np.zeros((2, 4), dtype=np.int32),
        })
        expected = np.stack([
            data_value[0, indices_value[0]],
            data_value[1, indices_value[1]],
        ])
        assert np.array_equal(got["gathered"], expected)
        assert np.array_equal(got["positions"], np.tile(np.arange(4, dtype=np.int32), (2, 1)))
        assert np.array_equal(got["seq_len"], np.array([4], dtype=np.int32))


def test_dynamic_batched_last_token_pooling_roundtrips_package(ctx, tmp_path):
    with Builder(ctx) as b:
        dynamic = {0: "batch", 1: "seq"}
        data = b.input((2, 4, 3), dynamic=dynamic).rename("data")
        tokens = b.input((2, 4), dtype=aion.int32, dynamic=dynamic).rename("tokens")
        mask = b.input((2, 4), dtype=aion.int32, dynamic=dynamic).rename("mask")
        lengths = b.reduce("sum", mask, axis=1)
        one = b.cast(b.constant(1.0), aion.int32)
        last = b.sub(lengths, one)
        pooled = b.squeeze(
            b.gather(data, b.unsqueeze(last, 1), axis=1, batch_dims=1),
            1,
        )
        path = tmp_path / "dynamic_pool.aion"
        b.export(str(path), {
            "pooled": pooled,
            "positions": b.iota(tokens, axis=1),
            "seq_len": b.dim(tokens, axis=1),
        })

    model = aion.load_model(str(path), thread_count=1)
    try:
        for batch, seq in ((1, 3), (3, 5)):
            data_value = np.arange(batch * seq * 3, dtype=np.float32).reshape(batch, seq, 3)
            lengths_value = np.arange(1, batch + 1, dtype=np.int32)
            lengths_value = np.minimum(lengths_value, seq)
            mask_value = (np.arange(seq)[None, :] < lengths_value[:, None]).astype(np.int32)
            got = model.run_numpy({
                "data": data_value,
                "tokens": np.zeros((batch, seq), dtype=np.int32),
                "mask": mask_value,
            })
            assert np.array_equal(got["pooled"], data_value[np.arange(batch), lengths_value - 1])
            assert np.array_equal(
                got["positions"],
                np.tile(np.arange(seq, dtype=np.int32), (batch, 1)),
            )
            assert np.array_equal(got["seq_len"], np.array([seq], dtype=np.int32))
    finally:
        model.close()


@pytest.mark.parametrize("op,fn", [
    ("eq", np.equal), ("ne", np.not_equal),
    ("lt", np.less), ("gt", np.greater),
    ("le", np.less_equal), ("ge", np.greater_equal),
])
def test_compare_i32(ctx, op, fn):
    with Builder(ctx) as b:
        a = b.input((2, 3), dtype=aion.int32).rename("a")
        c = b.input((2, 3), dtype=aion.int32).rename("c")
        y = b.compare(op, a, c).rename("y")
        m = b.compile({"y": y})
        av = np.array([[1, 5, 3], [4, 2, 6]], np.int32)
        cv = np.full((2, 3), 3, np.int32)
        got = m.run_numpy({"a": av, "c": cv})["y"]
        assert np.array_equal(got, fn(av, cv).astype(np.int32))


@pytest.mark.parametrize("method,pyfn", [
    ("mul", lambda x, v: x * v),
    ("sub", lambda x, v: x - v),
    ("div", lambda x, v: x / v),
])
def test_elemwise_suffix_broadcast(ctx, method, pyfn):
    with Builder(ctx) as b:
        x = b.input((2, 3)).rename("x")
        v = b.input((3,)).rename("v")
        y = getattr(b, method)(x, v).rename("y")
        m = b.compile({"y": y})
        xv = (np.arange(6, dtype=np.float32) + 1).reshape(2, 3)
        vv = np.array([1.0, 2.0, 4.0], np.float32)
        assert np.allclose(m.run({"x": xv, "v": vv})["y"], pyfn(xv, vv), atol=1e-6)

def test_general_right_aligned_broadcast(ctx):
    with Builder(ctx) as b:
        a = b.input((2, 1, 5)).rename("a")
        c = b.input((1, 3, 1)).rename("c")
        y = b.sub(a, c).rename("y")
        m = b.compile({"y": y})
        av = np.arange(10, dtype=np.float32).reshape(2, 1, 5)
        cv = np.array([[[1.0], [10.0], [100.0]]], dtype=np.float32)
        got = m.run({"a": av, "c": cv})["y"]
        assert got.shape == (2, 3, 5)
        assert np.array_equal(got, av - cv)


def test_reversed_scalar_i32_broadcast(ctx):
    with Builder(ctx) as b:
        scalar = b.input((1,), dtype=aion.int32).rename("scalar")
        values = b.input((2, 3), dtype=aion.int32).rename("values")
        quotient = b.div(scalar, values).rename("quotient")
        less = b.compare("lt", scalar, values).rename("less")
        m = b.compile({"quotient": quotient, "less": less})
        vv = np.array([[2, 0, -2], [5, -5, 1]], dtype=np.int32)
        got = m.run({"scalar": np.array([10], np.int32), "values": vv})
        expected = np.array([[5, 0, -5], [2, -2, 10]], dtype=np.int32)
        assert np.array_equal(got["quotient"], expected)
        assert np.array_equal(got["less"], (10 < vv).astype(np.int32))


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
