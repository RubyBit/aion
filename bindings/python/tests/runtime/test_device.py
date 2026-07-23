# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import aion
from aion.device import _device_to_str, _normalize_device
from aion.enums import AionDeviceKind


def _gpu_ctx_or_skip() -> aion.Context:
    """A single-GPU context, or skip if this build/machine has no usable GPU.

    Mirrors the Zig `test_api_gpu.zig` skip-on-`BackendUnavailable` pattern:
    a CPU-only build or a headless machine both surface as AION_UNSUPPORTED.
    """
    try:
        return aion.Context.gpu()
    except aion.AionError as e:
        pytest.skip(f"no usable GPU (build/adapter): {e}")


# --- Pure-Python device normalization (no native calls) ----------------------

def test_normalize_device_variants():
    cpu = int(AionDeviceKind.AION_DEVICE_CPU)
    gpu = int(AionDeviceKind.AION_DEVICE_GPU)
    assert _normalize_device(None) == (cpu, 0)
    assert _normalize_device("cpu") == (cpu, 0)
    assert _normalize_device("gpu") == (gpu, 0)
    assert _normalize_device("gpu:2") == (gpu, 2)
    assert _normalize_device(("gpu", 3)) == (gpu, 3)
    assert _normalize_device(AionDeviceKind.AION_DEVICE_GPU) == (gpu, 0)


def test_normalize_device_errors():
    with pytest.raises(ValueError):
        _normalize_device("cuda")
    with pytest.raises(ValueError):
        _normalize_device("gpu:-1")


def test_device_to_str():
    assert _device_to_str(int(AionDeviceKind.AION_DEVICE_CPU), 0) == "cpu"
    assert _device_to_str(int(AionDeviceKind.AION_DEVICE_GPU), 2) == "gpu:2"


# --- CPU-path behavior (always runs) -----------------------------------------

def test_cpu_tensor_reports_cpu_device():
    with aion.Context(thread_count=1) as ctx:
        with aion.Tensor([1.0, 2.0, 3.0], ctx=ctx) as t:
            assert t.device() == "cpu"


def test_to_gpu_without_registered_gpu_raises():
    # No GPU registered on this context -> migrating to gpu is an error, not a crash.
    with aion.Context(thread_count=1) as ctx:
        with aion.Tensor([1.0, 2.0], ctx=ctx) as t:
            with pytest.raises(aion.AionError):
                t.to("gpu")
            # Tensor stays host-resident and readable.
            assert t.device() == "cpu"
            assert t.tolist() == pytest.approx([1.0, 2.0])


def test_to_cpu_is_idempotent_noop():
    with aion.Context(thread_count=1) as ctx:
        with aion.Tensor([1.0, 2.0], ctx=ctx) as t:
            t.to("cpu")  # already on cpu -> no-op, must not raise
            assert t.device() == "cpu"


# --- GPU-path behavior (skipped without a usable GPU) ------------------------

def test_tensor_to_roundtrip_gpu():
    ctx = _gpu_ctx_or_skip()
    with ctx:
        vals = [float(i) - 3.0 for i in range(8)]
        with aion.Tensor(vals, ctx=ctx) as t:
            t.to("gpu")
            assert t.device() == "gpu:0"
            # Host reads are blocked while device-resident.
            with pytest.raises(aion.AionError):
                t.tolist()
            # Migrate back; bytes must survive the H2D + D2H round-trip.
            t.to("cpu")
            assert t.device() == "cpu"
            assert t.tolist() == pytest.approx(vals)


def test_tensor_construct_on_gpu():
    ctx = _gpu_ctx_or_skip()
    with ctx:
        with aion.Tensor([1.0, 2.0, 3.0, 4.0], ctx=ctx, device="gpu") as t:
            assert t.device() == "gpu:0"
            t.to("cpu")
            assert t.tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_model_load_on_gpu_matches_cpu_introspection(tiny_model_path):
    model_path = tiny_model_path
    with aion.Context(thread_count=1) as cpu_ctx:
        with aion.LoadedModel.load(cpu_ctx, str(model_path)) as m_cpu:
            cpu_ins = m_cpu.input_names()
            cpu_outs = m_cpu.output_names()

    ctx = _gpu_ctx_or_skip()
    with ctx:
        with aion.LoadedModel.load(ctx, str(model_path), device="gpu") as m_gpu:
            assert m_gpu.input_names() == cpu_ins
            assert m_gpu.output_names() == cpu_outs
