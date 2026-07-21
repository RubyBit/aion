# SPDX-License-Identifier: Apache-2.0
"""Symbolic dims in view ops (slice lens / reshape new_shape).

A slice/reshape dim may be a dim-symbol name (declared on an input via
`dynamic={axis: name}`); the exported model resolves it per run, serving any size
on that axis. Requires the C-ABI slice/reshape symbolic-length extension.
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


def test_symbolic_slice_and_reshape_export_serves_many_sizes(ctx, tmp_path):
    path = tmp_path / "sym.aion"
    with Builder(ctx) as b:
        x = b.input((2, 4), dynamic={0: "batch"}).rename("x")
        y = b.slice(x, (0, 1), ("batch", 2))     # symbolic len on the batch axis
        z = b.reshape(y, ("batch", 1, 2))        # symbolic reshape
        b.mark_output(z, "z")
        b.export(str(path), None)

    model = aion.load_model(str(path), thread_count=1)
    for bsz in (1, 2, 3, 5):
        xv = np.arange(bsz * 4, dtype=np.float32).reshape(bsz, 4)
        got = model.run({"x": xv})["z"]
        ref = xv[:, 1:3].reshape(bsz, 1, 2)
        assert got.shape == ref.shape
        assert np.allclose(got, ref)


def test_unknown_symbol_in_view_raises(ctx):
    with Builder(ctx) as b:
        x = b.input((2, 4)).rename("x")
        with pytest.raises(ValueError, match="unknown dim symbol"):
            b.reshape(x, ("nope", 4))


def test_concrete_slice_reshape_still_work(ctx):
    # All-constant dims take the non-symbolic path (backward compatible).
    with Builder(ctx) as b:
        x = b.input((4, 6)).rename("x")
        y = b.reshape(b.slice(x, (0, 0), (4, 4)), (4, 2, 2)).rename("y")
        m = b.compile({"y": y})
        xv = np.arange(24, dtype=np.float32).reshape(4, 6)
        assert np.allclose(m.run({"x": xv})["y"], xv[:, :4].reshape(4, 2, 2))
