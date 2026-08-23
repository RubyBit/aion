# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import build_zig  # noqa: E402


def test_run_retries_windows_child_exit_code_5_once(monkeypatch, tmp_path) -> None:
    results = iter(
        [
            subprocess.CompletedProcess(
                ["zig", "build"], 1, stderr="error: process exited with code 5\n"
            ),
            subprocess.CompletedProcess(["zig", "build"], 0, stderr=""),
        ]
    )
    calls = 0

    def fake_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        return next(results)

    monkeypatch.setattr(build_zig.platform, "system", lambda: "Windows")
    monkeypatch.setattr(build_zig.subprocess, "run", fake_run)
    monkeypatch.setattr(build_zig.time, "sleep", lambda _: None)

    build_zig._run(["zig", "build"], tmp_path)
    assert calls == 2


def test_run_does_not_retry_real_compiler_error(monkeypatch, tmp_path) -> None:
    calls = 0

    def fake_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            ["zig", "build"], 1, stderr="error: expected expression, found '}'\n"
        )

    monkeypatch.setattr(build_zig.platform, "system", lambda: "Windows")
    monkeypatch.setattr(build_zig.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        build_zig._run(["zig", "build"], tmp_path)
    assert calls == 1
