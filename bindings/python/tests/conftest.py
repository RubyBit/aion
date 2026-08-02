# SPDX-License-Identifier: Apache-2.0
"""Session fixtures: tiny synthetic `.aion` models built at test time.

Packages are authored with the C-ABI `aion.Builder` and serialized via
`.export()`, then loaded/executed by the Zig runtime — so every run exercises the
authoring path end to end, including the q8_0 quantized weight path (the q8
fixture doubles as a quant-layout parity canary against the numpy reference).
No checked-in model files, no network.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import aion
from aion import Builder


@dataclass(frozen=True)
class TinyModel:
    path: Path
    x: np.ndarray  # sample input
    w: np.ndarray  # dense weight [K, N] (f32 reference values)


def _build_tiny_f32(path: Path) -> TinyModel:
    """x [1,2,4] @ W [1,4,3] -> y [1,2,3], all f32."""
    w = (np.arange(12, dtype=np.float32).reshape(4, 3) * 0.25) - 1.0
    with Builder() as b:
        x = b.input((1, 2, 4)).rename("x")
        wp = b.param(w.reshape(1, 4, 3))
        y = b.matmul(x, wp).rename("y")
        b.add_metadata("arch", "test-tiny-matmul-f32")
        b.export(str(path), {"y": y})
    x_sample = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(1, 2, 4)
    return TinyModel(path=path, x=x_sample, w=w)


def _build_tiny_q8(path: Path) -> TinyModel:
    """x [2,32] @ W(q8_0) [32,3] -> y [2,3]; weights quantized q8_0 along K (axis 0).

    The core quantizer (`aion_tensor_quantize`) blocks along axis 0, the matmul-B
    `[K, N]` reduction axis — so the q8 weight is authored 2-D `[K, N]`.
    """
    rng = np.random.RandomState(7)
    w_torch = rng.randn(3, 32).astype(np.float32)  # torch [out, in] layout
    w = w_torch.T.copy()  # [K=32, N=3] reference
    with Builder() as b:
        x = b.input((2, 32)).rename("x")
        wp = b.param(w, dtype=aion.q8_0, shape=(32, 3))
        y = b.matmul(x, wp).rename("y")
        b.add_metadata("arch", "test-tiny-matmul-q8")
        b.export(str(path), {"y": y})
    x_sample = rng.randn(2, 32).astype(np.float32)
    return TinyModel(path=path, x=x_sample, w=w)


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory) -> TinyModel:
    return _build_tiny_f32(tmp_path_factory.mktemp("fixtures") / "tiny_matmul_f32.aion")


@pytest.fixture(scope="session")
def tiny_model_path(tiny_model) -> Path:
    return tiny_model.path


@pytest.fixture(scope="session")
def tiny_q8_model(tmp_path_factory) -> TinyModel:
    return _build_tiny_q8(tmp_path_factory.mktemp("fixtures") / "tiny_matmul_q8.aion")
