# SPDX-License-Identifier: Apache-2.0
"""DLPack both ways: host data read in place, and tensors exported without a copy."""
from __future__ import annotations

import numpy as np
import pytest

import aion

torch = pytest.importorskip("torch")


# --- in: any DLPack exporter, read where it lives -----------------------------


def test_torch_bf16_reads_in_place_everywhere():
    src = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.bfloat16)
    with aion.Context(thread_count=1) as ctx:
        with aion.Tensor(src, ctx=ctx) as t:
            assert t.dtype == aion.float32
            assert t.tolist() == [[0, 1, 2], [3, 4, 5]]
            t.copy_from(src * 2)
            assert t.tolist() == [[0, 2, 4], [6, 8, 10]]
        with aion.Tensor(src.T, ctx=ctx, dtype=aion.float16) as t:  # strided
            assert t.tolist() == [[0, 3], [1, 4], [2, 5]]
        q = aion.Tensor.quantize(ctx, (64, 32), torch.ones(64, 32, dtype=torch.bfloat16))
        assert q.shape == (64, 32)


def test_numpy_defaults_narrow_to_aion_dtypes():
    with aion.Context(thread_count=1) as ctx:
        with aion.Tensor(np.array([0.5, 1.5]), ctx=ctx) as t:  # float64
            assert t.dtype == aion.float32 and t.tolist() == [0.5, 1.5]
        with aion.Tensor(np.array([[1, 2]], dtype=np.int64), ctx=ctx) as t:
            assert t.dtype == aion.int32 and t.tolist() == [[1, 2]]
        with aion.Tensor.empty(ctx, (1,), dtype=aion.int32) as t:
            with pytest.raises(aion.AionError):  # does not fit: an error, never wrapped
                t.copy_from(np.array([1 << 40], dtype=np.int64))


def test_int_data_converts_into_a_float_tensor():
    with aion.Context(thread_count=1) as ctx:
        with aion.Tensor(np.arange(3), ctx=ctx, dtype=aion.float32) as t:
            assert t.tolist() == [0.0, 1.0, 2.0]
        with aion.Tensor([1, 2], ctx=ctx, dtype=aion.float16) as t:
            assert t.tolist() == [1.0, 2.0]


def test_a_single_element_broadcasts_and_other_shapes_do_not():
    with aion.Context(thread_count=1) as ctx:
        with aion.Tensor.empty(ctx, (2, 2)) as t:
            t.copy_from(np.float32(3.0))
            assert t.tolist() == [[3.0, 3.0], [3.0, 3.0]]
            t.copy_from(torch.tensor([7.0]))
            assert t.tolist() == [[7.0, 7.0], [7.0, 7.0]]
            with pytest.raises(ValueError, match="shape mismatch"):
                t.copy_from(np.zeros(4, np.float32))


def test_device_arrays_are_refused_up_front():
    class OnCuda:
        def __dlpack__(self, **_):  # pragma: no cover - must not be reached
            raise AssertionError("exported before the device check")

        def __dlpack_device__(self):
            return (2, 0)

    with pytest.raises(ValueError, match="CUDA"):
        aion.Tensor(OnCuda())


def test_protocol_type_matches_exporters():
    assert isinstance(np.zeros(1), aion.SupportsDLPack)
    assert isinstance(torch.zeros(1), aion.SupportsDLPack)
    with aion.Tensor([1.0]) as t:
        assert isinstance(t, aion.SupportsDLPack)
    assert not isinstance([1.0], aion.SupportsDLPack)


def test_data_already_in_the_tensor_dtype_is_borrowed_not_copied():
    arr = np.ones((2, 2), np.float32)
    with aion.from_dlpack(arr) as t:
        assert np.from_dlpack(t).ctypes.data == arr.ctypes.data  # the same bytes
        arr[0, 0] = 9.0  # the owner's write shows through
        assert t.tolist() == [[9.0, 1.0], [1.0, 1.0]]
        t.copy_from([[0.0, 0.0], [0.0, 0.0]])  # Aion's write goes to a private copy
        assert arr.tolist() == [[9.0, 1.0], [1.0, 1.0]]
        assert t.tolist() == [[0.0, 0.0], [0.0, 0.0]]
    src = torch.zeros(3)
    with aion.Tensor(src) as t:
        src[1] = 4.0
        assert t.tolist() == [0.0, 4.0, 0.0]
    with pytest.raises(TypeError, match="__dlpack__"):
        aion.from_dlpack([1.0])


def test_a_borrow_keeps_its_owner_alive_and_lets_go_once_done():
    import gc
    import weakref

    arr = np.arange(4, dtype=np.float32)
    alive = weakref.ref(arr)
    t = aion.Tensor(arr)
    exported = np.from_dlpack(t)
    del arr
    gc.collect()
    assert alive() is not None  # held by the tensor (and the export)
    t.close()
    gc.collect()
    assert alive() is not None and exported.tolist() == [0.0, 1.0, 2.0, 3.0]
    del exported
    gc.collect()
    assert alive() is None  # the last holder handed it back


def test_data_that_needs_converting_is_copied_once():
    with aion.Context(thread_count=1) as ctx:
        for data in (np.ones(3), np.ones((3, 2), np.float32).T, torch.ones(3, dtype=torch.bfloat16)):
            with aion.Tensor(data, ctx=ctx) as t:  # float64, transposed, bfloat16
                data[0] = 5
                assert np.asarray(t.tolist()).flat[0] == 1.0 and t.dtype == aion.float32
        with aion.Tensor(np.array(2.5, np.float32), ctx=ctx) as t:  # rank 0 -> (1,)
            assert t.shape == (1,) and t.tolist() == [2.5]


# --- out: a tensor's bytes, shared, read-only, never copied -------------------


def test_numpy_and_torch_view_a_tensor_without_copying():
    with aion.Context(thread_count=1) as ctx:
        t = aion.Tensor([[1.0, 2.0], [3.0, 4.0]], ctx=ctx)
        a = np.from_dlpack(t)
        b = np.from_dlpack(t)
        assert a.shape == (2, 2) and a.dtype == np.float32
        assert a.ctypes.data == b.ctypes.data  # the same bytes, no copy
        assert not a.flags.writeable
        assert torch.from_dlpack(t).tolist() == [[1.0, 2.0], [3.0, 4.0]]
        assert t.__dlpack_device__() == (1, 0)
        t.close()


def test_an_export_is_a_snapshot_that_outlives_the_tensor():
    with aion.Context(thread_count=1) as ctx:
        t = aion.Tensor([1.0, 2.0, 3.0], ctx=ctx)
        view = np.from_dlpack(t)
        t.copy_from([7.0, 8.0, 9.0])  # goes to a fresh copy while `view` holds the old bytes
        assert view.tolist() == [1.0, 2.0, 3.0]
        assert t.tolist() == [7.0, 8.0, 9.0]
        t.close()
        assert view.tolist() == [1.0, 2.0, 3.0]
        del view


def test_copy_true_exports_private_bytes():
    with aion.Tensor([1, 2, 3]) as t:
        a = np.from_dlpack(t)
        b = np.from_dlpack(t, copy=True)
        assert b.tolist() == [1, 2, 3]
        assert a.ctypes.data != b.ctypes.data


def test_an_unconsumed_capsule_frees_its_export():
    with aion.Tensor([1.0]) as t:
        for _ in range(3):
            capsule = t.__dlpack__(max_version=(1, 3))
            del capsule  # the capsule's destructor runs the deleter


def test_exports_refuse_what_they_cannot_share():
    with aion.Context(thread_count=1) as ctx:
        with aion.Tensor([1.0], ctx=ctx) as t:
            with pytest.raises(BufferError, match="versioned"):
                t.__dlpack__()
        q = aion.Tensor.quantize(ctx, (32, 1), [0.5] * 32)
        with pytest.raises(aion.AionError):
            np.from_dlpack(q)


def test_zero_clears_every_dtype():
    with aion.Context(thread_count=1) as ctx:
        for dtype in (aion.float32, aion.float16, aion.int32, aion.int8):
            with aion.Tensor([1, 2], ctx=ctx, dtype=dtype) as t:
                assert t.zero().tolist() == [0, 0]
