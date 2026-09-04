# SPDX-License-Identifier: Apache-2.0
"""PyTorch-like layers, mirroring the Zig `api.nn` catalog.

A layer holds its weights as `Parameter`s and composes ops in `forward`:

    class MLP(nn.Module):
        def __init__(self, w1, w2):
            self.fc1 = nn.Linear(w1)      # named "fc1" by the attribute
            self.fc2 = nn.Linear(w2)
        def forward(self, x):
            return self.fc2(self.fc1(x).relu())

    model = aion.compile(MLP(w1, w2), aion.spec((None, 4)))

Layers take and return `TensorRef` and read the builder off their input, so there is
no ambient state — `layer(x)` works against a bare `aion.Builder` too.

Parameters are named after the attribute path they were assigned at, so
`self.fc1 = nn.Linear(w)` binds `fc1/weight`. That is the name the package stores
and the name a loaded model looks the weight up by.

The stateful machinery behind this — scopes, parameter naming, constant caching —
lives once in the Zig `Builder` and is reached through the C ABI. What is written
in Python is only the per-layer composition.
"""
from __future__ import annotations

from .attention import Attention, AxialRoPE, KVCache, RelPosSelfAttention, RoPE
from .functional import scale, shift, softcap
from .layers import (
    Activation,
    Conv1D,
    Conv2D,
    DepthwiseConv1D,
    Embedding,
    LayerNorm,
    Linear,
    LSTMCell,
    RMSNorm,
)
from .mlp import FeedForward, GatedMLP, GLU
from .module import Module, ModuleList, Parameter, Residual, Sequential, builder_of

__all__ = [
    # infrastructure
    "Module",
    "ModuleList",
    "Parameter",
    "Residual",
    "Sequential",
    "builder_of",
    # projections / lookup
    "Linear",
    "Embedding",
    # normalization
    "LayerNorm",
    "RMSNorm",
    # convolution
    "Conv1D",
    "Conv2D",
    "DepthwiseConv1D",
    # recurrent
    "LSTMCell",
    # activations
    "Activation",
    # feed-forward blocks
    "FeedForward",
    "GatedMLP",
    "GLU",
    # attention
    "RoPE",
    "AxialRoPE",
    "KVCache",
    "Attention",
    "RelPosSelfAttention",
    # constant-bearing helpers
    "scale",
    "shift",
    "softcap",
]
