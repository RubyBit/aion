# SPDX-License-Identifier: Apache-2.0
"""Session fixtures: tiny synthetic `.aion` models built at test time.

The packages are written with `aion._writer` (the Python format encoder) and
loaded/executed by the Zig runtime, so every test run doubles as a
cross-implementation format-drift guard — including the q8_0 quantized path.
No checked-in model files, no network.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from aion._writer import Builder, format as aw


@dataclass(frozen=True)
class TinyModel:
    path: Path
    x: np.ndarray  # sample input [1, 2, K]
    w: np.ndarray  # dense weight [K, N] (f32 reference values)


def _build_tiny_f32(path: Path) -> TinyModel:
    """x [1,2,4] @ W [1,4,3] -> y [1,2,3], all f32."""
    b = Builder()
    x_id = b.add_input("x", aw.DType.f32, (1, 2, 4))
    w = (np.arange(12, dtype=np.float32).reshape(4, 3) * 0.25) - 1.0
    w_id = b.add_f32_initializer("w", w.reshape(1, 4, 3))
    b.add_output("y", b.matmul(x_id, w_id))
    b.pkg.metadata.append(aw.MetadataEntry(key="arch", value="test-tiny-matmul-f32"))
    b.write(str(path))
    x = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(1, 2, 4)
    return TinyModel(path=path, x=x, w=w)


def _build_tiny_q8(path: Path) -> TinyModel:
    """x [1,2,32] @ W(q8_0) [1,32,3] -> y [1,2,3]; weights quantized q8_0 along K."""
    b = Builder()
    x_id = b.add_input("x", aw.DType.f32, (1, 2, 32))
    rng = np.random.RandomState(7)
    w_torch = rng.randn(3, 32).astype(np.float32)  # torch [out, in] layout
    w_id = b.add_q8_0_matmul_b("w", w_torch)
    b.add_output("y", b.matmul(x_id, w_id))
    b.pkg.metadata.append(aw.MetadataEntry(key="arch", value="test-tiny-matmul-q8"))
    b.write(str(path))
    x = rng.randn(1, 2, 32).astype(np.float32)
    return TinyModel(path=path, x=x, w=w_torch.T.copy())


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory) -> TinyModel:
    return _build_tiny_f32(tmp_path_factory.mktemp("fixtures") / "tiny_matmul_f32.aion")


@pytest.fixture(scope="session")
def tiny_model_path(tiny_model) -> Path:
    return tiny_model.path


@pytest.fixture(scope="session")
def tiny_q8_model(tmp_path_factory) -> TinyModel:
    return _build_tiny_q8(tmp_path_factory.mktemp("fixtures") / "tiny_matmul_q8.aion")
