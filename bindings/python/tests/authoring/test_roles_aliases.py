# SPDX-License-Identifier: Apache-2.0
"""Input roles + output aliases: a recurrent-state model round-tripped through
export and driven by the runtime's auto-init / io-alias write-back."""
from __future__ import annotations

import numpy as np

import aion
from aion import Builder
from aion.enums import AionInputRoleKind as RK


def _build_counter(path) -> None:
    """state[1,4] (STATE role, zero-init) io-aliased to next_state = state + 1."""
    with Builder() as b:
        s = b.input((1, 4)).rename("state")
        one = b.param(np.ones((1, 4), np.float32))
        nxt = b.add(s, one)
        b.mark_output(nxt, "next_state")
        b.add_input_role(s, RK.AION_ROLE_STATE, zero_init=True)
        b.add_output_alias(s, nxt)
        b.export(str(path), None)


def test_state_role_and_alias_roundtrip(tmp_path):
    path = tmp_path / "counter.aion"
    _build_counter(path)
    assert path.exists() and path.stat().st_size > 0

    model = aion.load_model(str(path), thread_count=1)
    # Zero-init state, then io-alias write-back must accumulate across runs.
    r1 = model.run({}, outputs=["next_state"])["next_state"]
    r2 = model.run({}, outputs=["next_state"])["next_state"]
    r3 = model.run({}, outputs=["next_state"])["next_state"]
    assert np.allclose(r1, 1.0) and np.allclose(r2, 2.0) and np.allclose(r3, 3.0)

    # reset_state returns the aliased state to its zero-init value.
    model.reset_state()
    r4 = model.run({}, outputs=["next_state"])["next_state"]
    assert np.allclose(r4, 1.0)


def test_state_output_flagged_as_state(tmp_path):
    path = tmp_path / "counter2.aion"
    _build_counter(path)
    model = aion.load_model(str(path), thread_count=1)
    assert model.output_names() == ["next_state"]
    assert model.output_is_state(0) is True
