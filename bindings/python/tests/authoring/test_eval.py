# SPDX-License-Identifier: Apache-2.0
"""Evaluating a value while authoring, over the one graph.

`compile` hands the model a copy of the graph, so looking at a value costs nothing
structurally: the builder is untouched and authoring continues afterwards.
"""
import numpy as np
import pytest

import aion


@pytest.fixture()
def ctx():
    c = aion.Context(thread_count=1)
    try:
        yield c
    finally:
        c.close()


def _mm(b, ctx):
    a = aion.tensor(np.array([[1.0, -2.0]], np.float32), ctx=ctx)
    w = aion.tensor(np.array([[1.0, 0.0], [0.0, 1.0]], np.float32), ctx=ctx)
    return (b.param(a) @ b.param(w)).relu()


def test_eval_computes_a_value_mid_authoring(ctx):
    with aion.Builder(ctx) as b:
        y = _mm(b, ctx)
        assert np.allclose(y.numpy(), [[1.0, 0.0]])


def test_authoring_continues_after_eval(ctx):
    # The bug this guards: evaluating used to mark an output on the builder, so a
    # later value reused the first one's result.
    with aion.Builder(ctx) as b:
        y = _mm(b, ctx)
        assert np.allclose(y.numpy(), [[1.0, 0.0]])
        z = y + y
        assert np.allclose(z.numpy(), [[2.0, 0.0]])
        assert np.allclose(y.numpy(), [[1.0, 0.0]])  # unchanged


def test_compile_after_eval_yields_the_requested_output(ctx):
    with aion.Builder(ctx) as b:
        y = _mm(b, ctx)
        y.numpy()
        model = b.compile({"out": y + y})
        model.run()
        assert np.allclose(model.output_tensor("out").numpy(), [[2.0, 0.0]])


def test_compile_replaces_rather_than_accumulates_outputs(ctx):
    with aion.Builder(ctx) as b:
        y = _mm(b, ctx)
        m1 = b.compile({"a": y})
        m1.run()
        m2 = b.compile({"b": y + y})
        m2.run()
        # "a" was not carried into the second model.
        assert m2.output_names() == ["b"]


def test_repr_shows_the_computed_value(ctx):
    with aion.Builder(ctx) as b:
        y = _mm(b, ctx)
        text = repr(y)
        assert "shape=(1, 2)" in text and "1.0" in text


def test_eval_refuses_when_an_input_has_no_data(ctx):
    # The runtime would zero-fill an unbound input, so evaluating would hand back
    # numbers that look real. Refuse, and name the input.
    with aion.Builder(ctx) as b:
        x = b.input((1, 2)).rename("x")
        with pytest.raises(ValueError, match="no data bound.*x"):
            x.relu().numpy()


def test_repr_of_an_uncomputable_value_still_describes_it(ctx):
    # repr must degrade to shape/dtype rather than raise.
    with aion.Builder(ctx) as b:
        x = b.input((1, 2)).rename("x")
        text = repr(x.relu())
        assert "shape=(1, 2)" in text and "value=" not in text


# --- authoring without naming a Builder --------------------------------------
def test_ref_uses_the_context_scratch_builder(ctx):
    x = aion.tensor(np.array([[1.0, -2.0]], np.float32), ctx=ctx)
    w = aion.tensor(np.array([[1.0, 0.0], [0.0, 1.0]], np.float32), ctx=ctx)

    y = (x.ref() @ w.ref()).relu()
    assert np.allclose(y.numpy(), [[1.0, 0.0]])


def test_scratch_builder_is_reused(ctx):
    assert ctx.scratch_builder() is ctx.scratch_builder()


def test_nn_layers_compose_into_the_scratch_graph(ctx):
    from aion import nn

    x = aion.tensor(np.array([[1.0, -2.0]], np.float32), ctx=ctx)
    eye = aion.tensor(np.eye(2, dtype=np.float32), ctx=ctx)

    y = nn.Linear(eye)(x.ref()).relu()
    assert np.allclose(y.numpy(), [[1.0, 0.0]])


def test_ref_honours_an_explicit_builder(ctx):
    w = aion.tensor(np.eye(2, dtype=np.float32), ctx=ctx)
    with aion.Builder(ctx) as b:
        assert w.ref(b)._b is b


def test_ref_carries_the_rename_as_the_parameter_name(ctx):
    # The parameter name is how a loaded model looks a weight up.
    w = aion.tensor(np.eye(2, dtype=np.float32), ctx=ctx).rename("fc/weight")
    with aion.Builder(ctx) as b:
        ref = w.ref(b)
        assert b.value_name(ref) == "fc/weight"
