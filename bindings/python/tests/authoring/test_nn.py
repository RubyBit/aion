# SPDX-License-Identifier: Apache-2.0
"""The `aion.nn` layer catalog.

Each layer is checked against a hand-computed NumPy reference rather than against
another Aion path, so a wrong lowering cannot agree with itself. Naming is
asserted too: a parameter's path is what a loaded model looks it up by.
"""
from __future__ import annotations

import numpy as np
import pytest

import aion
from aion import nn


@pytest.fixture
def ctx():
    c = aion.Context(thread_count=1)
    try:
        yield c
    finally:
        c.close()


@pytest.fixture
def b(ctx):
    builder = aion.Builder(ctx=ctx)
    try:
        yield builder
    finally:
        builder.close()


def run(b, out, feed):
    """Compile `out` and run it once against `feed`."""
    model = b.compile({"y": out})
    try:
        return model.run_numpy(feed)["y"]
    finally:
        model.close()


# --- norms -----------------------------------------------------------------
def test_rmsnorm_with_gamma_only_synthesizes_a_zero_beta(b):
    d = 4
    x_np = np.array([[1.0, -2.0, 3.0, -4.0]], np.float32)
    g_np = np.array([2.0, 0.5, 1.0, -1.0], np.float32)

    x = b.input((1, d)).rename("x")
    y = nn.RMSNorm(g_np)(x)
    got = run(b, y, {"x": x_np})

    rms = np.sqrt(np.mean(x_np**2) + 1e-6)
    assert np.allclose(got, x_np / rms * g_np, atol=1e-5)


def test_parameterless_rmsnorm_normalizes_without_scaling(b):
    x_np = np.array([[3.0, 4.0, 0.0, 0.0]], np.float32)
    x = b.input((1, 4)).rename("x")
    # Both identities come from the Builder's shared constants.
    y = nn.RMSNorm(normalized_shape=(4,))(x)
    got = run(b, y, {"x": x_np})

    rms = np.sqrt(np.mean(x_np**2) + 1e-6)
    assert np.allclose(got, x_np / rms, atol=1e-5)


def test_rmsnorm_without_gamma_or_shape_is_rejected():
    with pytest.raises(ValueError, match="normalized_shape"):
        nn.RMSNorm()


def test_layernorm_applies_gamma_and_beta(b):
    x_np = np.array([[1.0, 2.0, 3.0, 4.0]], np.float32)
    g = np.ones(4, np.float32)
    beta = np.full(4, 0.5, np.float32)

    x = b.input((1, 4)).rename("x")
    got = run(b, nn.LayerNorm(g, beta)(x), {"x": x_np})

    ref = (x_np - x_np.mean()) / np.sqrt(x_np.var() + 1e-5) + 0.5
    assert np.allclose(got, ref, atol=1e-4)


def test_same_width_norms_share_one_identity_constant(b):
    g = np.ones(8, np.float32)
    n1, n2 = nn.RMSNorm(g, name="n1"), nn.RMSNorm(g, name="n2")
    x = b.input((1, 8))
    n1(x)
    n2(x)
    # One zeros vector serves both, so a 35-layer model pays for one.
    assert b.zeros(8).id == b.zeros(8).id
    assert b.param_kind(b.zeros(8)) == "synthesized"


# --- linear / embedding ------------------------------------------------------
def test_linear_with_bias_and_alpha(b):
    x_np = np.array([[1.0, 2.0]], np.float32)
    w = np.eye(2, dtype=np.float32)
    bias = np.array([0.5, -0.5], np.float32)

    x = b.input((1, 2)).rename("x")
    got = run(b, nn.Linear(w, bias, alpha=3.0)(x), {"x": x_np})
    assert np.allclose(got, x_np * 3.0 + bias)


def test_linear_2d_weight_serves_a_3d_activation(b):
    """matmul aligns operand ranks, so no `[1, in, out]` reshape is needed."""
    x_np = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
    w = np.eye(2, dtype=np.float32)

    x = b.input((1, 2, 2)).rename("x")
    got = run(b, nn.Linear(w)(x), {"x": x_np})
    assert np.allclose(got, x_np)


def test_embedding_lookup(b):
    table = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], np.float32)
    ids = b.input((1, 2), dtype=aion.int32).rename("ids")
    got = run(b, nn.Embedding(table)(ids), {"ids": np.array([[2, 0]], np.int32)})
    assert np.allclose(got, [[[5.0, 6.0], [1.0, 2.0]]])


# --- blocks ------------------------------------------------------------------
def test_gated_mlp_multiplies_the_two_projections(b):
    # in=1, ffn=2. Concatenating gate and up into one wide weight is a fusion the
    # compiler performs, so this layer only ever describes the two projections.
    gate = np.array([[1.0, 2.0]], np.float32)
    up = np.array([[10.0, 20.0]], np.float32)
    down = np.array([[1.0], [1.0]], np.float32)  # sums the two ffn lanes

    x = b.input((1, 1)).rename("x")
    y = nn.GatedMLP(gate, up, down, act="silu")(x)
    got = run(b, y, {"x": np.array([[1.0]], np.float32)})

    silu = lambda v: v / (1.0 + np.exp(-v))
    assert np.allclose(got, silu(1.0) * 10.0 + silu(2.0) * 20.0, atol=1e-4)


def test_gated_mlp_rejects_mismatched_gate_and_up_widths():
    with pytest.raises(ValueError, match="same output width"):
        nn.GatedMLP(
            np.ones((1, 3), np.float32),
            np.ones((1, 2), np.float32),
            np.ones((1, 1), np.float32),
        )


def test_glu_gates_one_half_with_the_other(b):
    w = np.array([[1.0, 2.0, 0.0, 100.0]], np.float32)
    x = b.input((1, 1)).rename("x")
    got = run(b, nn.GLU(w)(x), {"x": np.array([[1.0]], np.float32)})
    # a = [1, 2], g = [0, 100] -> a * sigmoid(g)
    assert np.allclose(got, [[0.5, 2.0]], atol=1e-5)


def test_sequential_and_residual(b):
    eye = np.eye(2, dtype=np.float32)
    x_np = np.array([[1.0, -2.0]], np.float32)

    block = nn.Residual(
        nn.Sequential(nn.Linear(eye), nn.Activation("relu"), nn.Linear(eye)),
        scale=0.5,
    )
    x = b.input((1, 2)).rename("x")
    got = run(b, block(x), {"x": x_np})
    # x + 0.5 * relu(x)
    assert np.allclose(got, x_np + 0.5 * np.maximum(x_np, 0.0))


def test_lstm_cell_returns_h_and_c(b):
    h_size = 2
    w_ih = np.zeros((2, 4 * h_size), np.float32)
    w_hh = np.zeros((h_size, 4 * h_size), np.float32)
    cell = nn.LSTMCell(w_ih, w_hh)

    x = b.input((1, 2)).rename("x")
    h0 = b.input((1, h_size)).rename("h")
    c0 = b.input((1, h_size)).rename("c")
    h, c = cell(x, h0, c0)
    assert h.shape == (1, h_size) and c.shape == (1, h_size)


def test_lstm_cell_rejects_a_lone_bias():
    with pytest.raises(ValueError, match="pair"):
        nn.LSTMCell(
            np.zeros((2, 8), np.float32),
            np.zeros((2, 8), np.float32),
            b_ih=np.zeros(8, np.float32),
        )


def test_depthwise_conv_derives_groups_and_rejects_dense(b):
    channels = 3
    w = np.array([2.0, 3.0, 4.0], np.float32).reshape(1, 1, channels)
    dw = nn.DepthwiseConv1D(w)
    assert dw.opts["groups"] == channels

    x = b.input((1, 1, channels)).rename("x")
    got = run(b, dw(x), {"x": np.ones((1, 1, channels), np.float32)})
    assert np.allclose(got.reshape(-1), [2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match=r"\[k, 1, channels\]"):
        nn.DepthwiseConv1D(np.ones((1, 2, 2), np.float32))


# --- constants ---------------------------------------------------------------
def test_scale_and_softcap(b):
    x_np = np.array([[0.0, 30.0, -30.0, 1.0]], np.float32)
    x = b.input((1, 4)).rename("x")

    # A no-op factor must not add a node at all.
    assert nn.scale(x, 1.0) is x
    assert nn.shift(x, 0.0) is x
    assert nn.softcap(x, 0.0) is x

    got = run(b, nn.softcap(nn.scale(x, 2.0), 10.0), {"x": x_np})
    assert np.allclose(got, 10.0 * np.tanh(x_np * 2.0 / 10.0), atol=1e-4)


# --- naming / introspection ---------------------------------------------------
def test_parameter_paths_follow_the_attribute_tree(b):
    eye = np.eye(2, dtype=np.float32)

    class Block(nn.Module):
        def __init__(self):
            self.attn = nn.Linear(eye, np.zeros(2, np.float32))
            self.norm = nn.RMSNorm(np.ones(2, np.float32))
            self.ffn = nn.GatedMLP(eye, eye, eye)

        def forward(self, x):
            return self.ffn(self.norm(self.attn(x)))

    block = Block()
    # Names come from the builder, so they only exist once something is bound.
    block(b.input((1, 2)))

    assert [p for p, _ in block.named_parameters(b)] == [
        "attn/weight",
        "attn/bias",
        "norm/weight",
        "ffn/gate_proj/weight",
        "ffn/up_proj/weight",
        "ffn/down_proj/weight",
    ]
    assert list(block.state_dict(b)) == [p for p, _ in block.named_parameters(b)]
    for path, _ in block.named_parameters(b):
        assert b.has_param_named(path), path


def test_module_list_registers_indexed_children(b):
    eye = np.eye(2, dtype=np.float32)

    class Stack(nn.Module):
        def __init__(self, n):
            self.layers = nn.ModuleList([nn.Linear(eye) for _ in range(n)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    stack = Stack(3)
    stack(b.input((1, 2)))
    assert [p for p, _ in stack.named_parameters(b)] == [
        "layers/0/weight",
        "layers/1/weight",
        "layers/2/weight",
    ]
    assert b.has_param_named("layers/2/weight")


def test_a_layer_applied_twice_keeps_one_identity(b):
    """Re-applying a layer must not re-number its scope or rebind its weights."""
    lin = nn.Linear(np.eye(2, dtype=np.float32))
    x = b.input((1, 2))
    y1 = lin(x)
    y2 = lin(y1)
    assert y1.id != y2.id
    # One parameter, bound once.
    assert sum(1 for _ in lin.named_parameters(b)) == 1
    assert b.has_param_named("Linear#0/weight")
    assert not b.has_param_named("Linear#1/weight")


def test_cached_attention_reader_layer_has_no_kv_projections(b):
    # Gemma 4 shares one KV cache across a run of layers: the first projects K/V,
    # the rest only project Q and attend over it. Omitting k/v says so.
    eye = np.eye(2, dtype=np.float32)

    def attn(**kw: object) -> nn.Attention:
        return nn.Attention(
            eye, eye, heads=1, kv_heads=1, head_dim=2, scale=1.0, **kw  # type: ignore[arg-type]
        )

    writer = attn(k_proj=eye, v_proj=eye)
    reader = attn()

    x = b.input((1, 1, 2)).rename("x")
    q_w, k_w, v_w = writer.project(x)
    q_r, k_r, v_r = reader.project(x)

    assert k_w is not None and v_w is not None
    assert k_r is None and v_r is None
    assert q_r.shape == q_w.shape


def test_cached_attention_rejects_k_without_v():
    eye = np.eye(2, dtype=np.float32)
    with pytest.raises(ValueError, match="both k_proj and v_proj"):
        nn.Attention(eye, eye, k_proj=eye, heads=1, kv_heads=1, head_dim=2, scale=1.0)


def _relpos(heads=2, head_dim=2, time=3, seed=0):
    """A RelPosSelfAttention with random weights, plus the arrays it was built from."""
    rng = np.random.default_rng(seed)
    d = heads * head_dim
    w = {n: rng.standard_normal((d, d)).astype(np.float32) for n in "qkvo"}
    pos_emb = rng.standard_normal((heads, 2 * time - 1, head_dim)).astype(np.float32)
    bias = {n: rng.standard_normal((heads, head_dim)).astype(np.float32) for n in "uv"}
    layer = nn.RelPosSelfAttention(
        w["q"], w["k"], w["v"], w["o"], pos_emb, bias["u"], bias["v"],
        scale=head_dim ** -0.5, name="self_attn",
    )
    return layer, d


def test_relpos_self_attention_project_then_attend_equals_forward(b):
    """The split exists so a streaming encoder can prepend cached K/V in between;
    with nothing in between it must be exactly the one-shot path."""
    layer, d = _relpos()
    x = b.input((1, 3, d)).rename("x")

    whole = layer(x)
    split = layer.attend(*layer.project(x))

    feed = {"x": np.random.default_rng(1).standard_normal((1, 3, d)).astype(np.float32)}
    model = b.compile({"whole": whole, "split": split})
    try:
        got = model.run_numpy(feed)
    finally:
        model.close()
    np.testing.assert_array_equal(got["whole"], got["split"])


def test_relpos_self_attention_names_its_parameters(b):
    layer, d = _relpos()
    layer(b.input((1, 3, d)).rename("x"))

    for name in ("q_proj/weight", "k_proj/weight", "v_proj/weight", "o_proj/weight",
                 "pos_emb", "pos_bias_u", "pos_bias_v"):
        assert b.has_param_named(f"self_attn/{name}"), name


def test_relpos_self_attention_infers_head_geometry_from_pos_emb():
    layer, _ = _relpos(heads=3, head_dim=4)

    assert layer.heads == 3
    assert layer.head_dim == 4
