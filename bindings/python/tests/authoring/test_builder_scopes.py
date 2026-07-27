# SPDX-License-Identifier: Apache-2.0
"""Scopes, named params, and synthesized constants on the Builder.

These are Zig `Builder` features surfaced through the C ABI, not reimplemented
here — so what these assert is that the binding reaches the one implementation,
and that its behaviour matches `src/aion/api/test_api.zig`.
"""
from __future__ import annotations

import numpy as np
import pytest

import aion


@pytest.fixture
def b(ctx):
    builder = aion.Builder(ctx=ctx)
    try:
        yield builder
    finally:
        builder.close()


@pytest.fixture
def ctx():
    c = aion.Context(thread_count=1)
    try:
        yield c
    finally:
        c.close()


def test_scopes_nest_and_qualify_parameter_names(b):
    assert b.scope_path == ""
    with b.scope("layers.3"):
        assert b.scope_path == "layers.3"
        with b.scope("q_proj"):
            assert b.scope_path == "layers.3/q_proj"
            w = b.param_named(np.eye(2, dtype=np.float32), "weight")
    # Closed again once the blocks exit.
    assert b.scope_path == ""
    assert b.has_param_named("layers.3/q_proj/weight")
    assert b.param_kind(w) == "user"


def test_scope_closes_even_when_the_body_raises(b):
    with pytest.raises(RuntimeError):
        with b.scope("outer"):
            raise RuntimeError("boom")
    assert b.scope_path == ""


def test_auto_scope_numbers_siblings_per_parent(b):
    seen = []
    for _ in range(2):
        with b.auto_scope("Block"):
            for _ in range(2):
                with b.auto_scope("Lin"):
                    seen.append(b.scope_path)

    # Inner counters restart under each parent.
    assert seen == [
        "Block#0/Lin#0",
        "Block#0/Lin#1",
        "Block#1/Lin#0",
        "Block#1/Lin#1",
    ]


def test_constants_are_cached_and_scope_independent(b):
    z1, o1, k1 = b.zeros(4), b.ones(4), b.constant(0.5)

    with b.scope("block"):
        # Same value from inside a scope: still the one constant, because a shared
        # constant belongs to no single layer.
        assert b.zeros(4).id == z1.id
        assert b.ones(4).id == o1.id
        assert b.constant(0.5).id == k1.id

    assert b.zeros(8).id != z1.id
    assert b.constant(0.25).id != k1.id
    # One element regardless of what it will multiply.
    assert k1.shape == (1,)
    assert z1.shape == (4,)


def test_constants_are_marked_synthesized_not_model_state(b):
    w = b.param_named(np.eye(2, dtype=np.float32), "weight")
    z = b.zeros(2)

    assert b.param_kind(w) == "user"
    assert b.param_kind(z) == "synthesized"
    # A value that is not a parameter at all.
    assert b.param_kind(b.input((1, 2))) is None


def test_param_named_rejects_an_empty_name(b):
    with pytest.raises(aion.AionError):
        b.param_named(np.eye(2, dtype=np.float32), "")


def test_named_params_survive_export_and_load(ctx, tmp_path):
    """The names reach the package, which is what load-by-name depends on."""
    path = str(tmp_path / "named.aion")

    b = aion.Builder(ctx=ctx)
    try:
        x = b.input((1, 2)).rename("x")
        with b.scope("fc"):
            w = b.param_named(np.eye(2, dtype=np.float32), "weight")
        y = (x @ w).rename("y")
        b.export(path, {"y": y})
    finally:
        b.close()

    m = aion.load_model(path)
    try:
        got = m.run_numpy({"x": np.array([[1.0, 2.0]], np.float32)})["y"]
        assert np.allclose(got, [[1.0, 2.0]])
    finally:
        m.close()
