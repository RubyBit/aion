# SPDX-License-Identifier: Apache-2.0
"""Host memory as DLPack: the one form data crosses the C ABI in, both ways.

In, `host_view` is the single funnel: anything exporting ``__dlpack__`` (numpy,
PyTorch, JAX, ...; DLPack 1.x) is viewed where it lives, strided and in its own
dtype, and Python numbers and nested sequences are packed into one float64 or
int64 buffer. A new tensor borrows an exporter's memory outright when it already
is what the tensor needs; otherwise, and for a write into an existing tensor, the
core converts in one pass (floats among floats, integers among integers). Only a
cross-kind conversion (int data into a float tensor, say) goes through numpy.

Out, `export_capsule` wraps a tensor's DLPack export in the capsule any consumer
(``np.from_dlpack``, ``torch.from_dlpack``) takes.
"""
from __future__ import annotations

import contextlib
import ctypes
import math
import numbers
from collections.abc import Generator, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

from ..enums import AionDType
from ._raw import ffi, lib

if TYPE_CHECKING:
    from typing_extensions import CapsuleType

_capsule_is_valid = ctypes.pythonapi.PyCapsule_IsValid
_capsule_is_valid.restype = ctypes.c_int
_capsule_is_valid.argtypes = [ctypes.py_object, ctypes.c_char_p]
_capsule_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_capsule_pointer.restype = ctypes.c_void_p
_capsule_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_capsule_set_name = ctypes.pythonapi.PyCapsule_SetName
_capsule_set_name.restype = ctypes.c_int
_capsule_set_name.argtypes = [ctypes.py_object, ctypes.c_char_p]
_capsule_new = ctypes.pythonapi.PyCapsule_New
_capsule_new.restype = ctypes.py_object
_capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

# Capsule names (module-level, so the bytes a capsule points at outlive it). A
# consumer that takes ownership renames the capsule `used_...`, and the capsule's
# destructor then leaves the managed tensor alone.
_VERSIONED = b"dltensor_versioned"
_USED = b"used_dltensor_versioned"
# The C destructor compiled into the extension (`_ffi/build.py`).
_DESTRUCTOR = int(ffi.cast("uintptr_t", ffi.addressof(lib, "aionpy_dlpack_capsule_destructor")))

_READ_ONLY = 1  # DLPACK_FLAG_BITMASK_READ_ONLY
_DEVICE_NAMES = {2: "CUDA", 4: "OpenCL", 7: "Vulkan", 8: "Metal", 10: "ROCm", 15: "WebGPU"}

Kind = Literal["float", "int"]

_DTYPE_KINDS: dict[AionDType, Kind] = {
    AionDType.AION_DTYPE_F32: "float",
    AionDType.AION_DTYPE_F16: "float",
    AionDType.AION_DTYPE_I32: "int",
    AionDType.AION_DTYPE_I8: "int",
}


def dtype_kind(dtype: AionDType) -> Kind:
    """Whether a scalar tensor dtype holds floats or integers."""
    try:
        return _DTYPE_KINDS[AionDType(int(dtype))]
    except KeyError:
        raise ValueError(f"{AionDType(int(dtype)).name} has no host element form") from None


class HostView:
    """A `DLTensor*` to hand the C ABI, and what keeps the memory it names alive."""

    __slots__ = ("ptr", "read_only", "_owner", "_capsule", "_managed")

    def __init__(
        self,
        ptr: Any,
        owner: object,
        *,
        read_only: bool = False,
        capsule: object = None,
        managed: Any = None,
    ) -> None:
        self.ptr = ptr
        self.read_only = read_only
        self._owner = owner
        self._capsule = capsule
        self._managed = managed

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(self.ptr.shape[i]) for i in range(int(self.ptr.ndim)))

    @property
    def kind(self) -> Kind | None:
        """Floats or integers the core reads, or None for a dtype it does not."""
        t = self.ptr.dtype
        code, bits = int(t.code), int(t.bits)
        if int(t.lanes) != 1:
            return None
        if (code == lib.kDLFloat and bits in (64, 32, 16)) or (code == lib.kDLBfloat and bits == 16):
            return "float"
        if code == lib.kDLInt and bits in (64, 32, 8):
            return "int"
        return None

    def fit(self, shape: Sequence[int], *, reshape: bool = False) -> HostView:
        """This view as `shape`: itself when it already is, or one element broadcast
        (zero strides); with `reshape`, also a contiguous view of the same count
        re-dimensioned (flat values for an explicit shape)."""
        shp = tuple(int(d) for d in shape)
        have = self.shape
        if have == shp:
            return self
        count = math.prod(have)
        if count == 1:
            return self._redim(shp, broadcast=True)
        if reshape and count == math.prod(shp) and self._contiguous():
            return self._redim(shp, broadcast=False)
        raise ValueError(f"expected shape {shp}, got {have}")

    def _contiguous(self) -> bool:
        if self.ptr.strides == ffi.NULL:
            return True
        want = 1
        for axis in reversed(range(int(self.ptr.ndim))):
            d = int(self.ptr.shape[axis])
            if d != 1 and int(self.ptr.strides[axis]) != want:
                return False
            want *= d
        return True

    def _redim(self, shape: tuple[int, ...], *, broadcast: bool) -> HostView:
        dims = ffi.new("int64_t[]", shape) if shape else ffi.NULL
        strides = ffi.new("int64_t[]", len(shape)) if broadcast and shape else ffi.NULL  # zeros
        dl = ffi.new("DLTensor*")
        dl[0] = self.ptr[0]
        dl.ndim = len(shape)
        dl.shape = dims
        dl.strides = strides
        return HostView(dl, (self, dims, strides), read_only=self.read_only or broadcast)

    @property
    def needs_keepalive(self) -> bool:
        """After `transfer`, whether the caller must keep this view alive: true for
        one of our own buffers, whose wrapper has no deleter to tell it the core is
        done."""
        return self._managed is None

    @contextlib.contextmanager
    def transfer(self) -> Generator[Any, None, None]:
        """Hand the core ownership: yields a `DLManagedTensorVersioned*` for one C
        call that takes it on success. A producer's own managed tensor goes as it
        is and is marked consumed when the block exits normally, so its capsule no
        longer frees it; if the call raises, the capsule keeps it. One of our own
        buffers goes as a wrapper with no deleter, alive as long as this view."""
        if self._managed is None:
            wrapper = ffi.new("DLManagedTensorVersioned*")
            wrapper.version.major = 1
            wrapper.version.minor = 3
            wrapper.deleter = ffi.NULL
            wrapper.dl_tensor = self.ptr[0]
            self._owner = (self._owner, wrapper)
            yield wrapper
            return
        yield self._managed
        if _capsule_set_name(self._capsule, _USED) != 0:
            raise RuntimeError("could not mark a DLPack capsule consumed")


def host_view(data: object, dtype: AionDType | None = None) -> HostView:
    """`data` as a view the core reads, for a tensor of `dtype` (or its own).

    A ``__dlpack__`` exporter is viewed in place, without copying. Python numbers
    and nested sequences are packed into one float64 buffer (int64 when every value
    is an integer, or `dtype` is an integer dtype). Data of the other kind than
    `dtype`, or of a dtype the core does not read, is converted by numpy first.
    """
    want = dtype_kind(dtype) if dtype is not None else None
    if hasattr(data, "__dlpack__"):
        view = _view_of(data)
        if view.kind is not None and want in (None, view.kind):
            return view
        if dtype is None:
            raise TypeError(f"{type(data).__name__} holds a dtype Aion does not read; pass dtype=")
        return _view_of(_numpy_converted(data, dtype))
    shape, flat = _flatten(data)
    floating = want == "float" if want is not None else any(isinstance(v, float) for v in flat)
    return _buffer(shape, "float" if floating else "int", flat)[1]


def empty_view(shape: Sequence[int], kind: Kind) -> tuple[Any, HostView]:
    """A zeroed float64 or int64 buffer of `shape` and a writable view of it, for
    reading a tensor out into Python numbers."""
    shp = tuple(int(d) for d in shape)
    return _buffer(shp, kind, None)


def data_shape(data: object) -> tuple[int, ...]:
    """The shape of weight or tensor data, without viewing or copying it."""
    shape = getattr(data, "shape", None)
    if shape is not None and not callable(shape):
        return tuple(int(d) for d in shape)
    return _flatten(data)[0]


def export_capsule(managed: Any) -> CapsuleType:
    """A ``dltensor_versioned`` capsule owning `managed`: a consumer takes it over,
    and one nobody consumed runs its deleter when collected."""
    return cast("CapsuleType", _capsule_new(int(ffi.cast("uintptr_t", managed)), _VERSIONED, _DESTRUCTOR))


def _view_of(obj: Any) -> HostView:
    """View `obj` through its ``__dlpack__`` export, without copying.

    Until it is handed over (`HostView.transfer`), the capsule stays unconsumed,
    so dropping the view lets the capsule's destructor release the producer's
    memory, as the protocol specifies.
    """
    device = getattr(obj, "__dlpack_device__", None)
    kind = int(device()[0]) if device is not None else lib.kDLCPU
    if kind != lib.kDLCPU:
        where = _DEVICE_NAMES.get(kind, f"device type {kind}")
        raise ValueError(f"{type(obj).__name__} lives on {where}; copy it to host memory first")
    capsule = obj.__dlpack__(max_version=(1, 3))
    if not _capsule_is_valid(capsule, _VERSIONED):
        raise TypeError(f"{type(obj).__name__}.__dlpack__ did not return a DLPack 1.x (versioned) capsule")
    versioned = ffi.cast("DLManagedTensorVersioned*", _capsule_pointer(capsule, _VERSIONED))
    if int(versioned.version.major) != 1:
        raise ValueError(f"DLPack major version {int(versioned.version.major)} is not supported (need 1)")
    dl = ffi.addressof(versioned[0], "dl_tensor")
    return HostView(dl, (obj, capsule), read_only=bool(versioned.flags & _READ_ONLY), capsule=capsule, managed=versioned)


def _numpy_converted(data: object, dtype: AionDType) -> Any:
    try:
        import numpy as np
    except ImportError:
        raise TypeError(
            f"{type(data).__name__} needs converting to {AionDType(int(dtype)).name}, which takes numpy"
        ) from None
    from ..dtype import numpy_dtype

    return np.asarray(data).astype(numpy_dtype(dtype))


def _flatten(data: object) -> tuple[tuple[int, ...], list[int | float]]:
    """Shape and row-major values of a Python number or nested sequence."""
    if isinstance(data, bool):
        return (), [int(data)]
    if isinstance(data, numbers.Integral):
        return (), [int(data)]
    if isinstance(data, numbers.Real):
        return (), [float(data)]
    if isinstance(data, (str, bytes)):
        raise TypeError("tensor data must be numbers, not a string")
    try:
        items = list(cast(Iterable[object], data))
    except TypeError:
        raise TypeError(f"tensor data must be a number, a nested sequence, or a DLPack exporter, not {type(data).__name__}") from None
    if not items:
        return (0,), []
    first, values = _flatten(items[0])
    for item in items[1:]:
        shape, more = _flatten(item)
        if shape != first:
            raise ValueError(f"ragged nested sequence: {shape} != {first}")
        values.extend(more)
    return (len(items),) + first, values


def _buffer(shape: tuple[int, ...], kind: Kind, values: list[int | float] | None) -> tuple[Any, HostView]:
    count = math.prod(shape)
    if values is not None and len(values) != count:
        raise ValueError(f"expected {count} values for shape {shape}, got {len(values)}")
    # At least one element: an empty buffer's address may be NULL, which DLPack
    # reserves for no data at all.
    ctype = "double" if kind == "float" else "int64_t"
    buf = ffi.new(f"{ctype}[]", max(count, 1))
    if values:
        buf[0:count] = [float(v) for v in values] if kind == "float" else [int(v) for v in values]
    dims = ffi.new("int64_t[]", shape) if shape else ffi.NULL
    dl = ffi.new("DLTensor*")
    dl.data = ffi.cast("void*", buf)
    dl.device.device_type = lib.kDLCPU
    dl.ndim = len(shape)
    dl.dtype.code = lib.kDLFloat if kind == "float" else lib.kDLInt
    dl.dtype.bits = 64
    dl.dtype.lanes = 1
    dl.shape = dims
    dl.strides = ffi.NULL
    return buf, HostView(dl, (buf, dims))


__all__ = ["HostView", "data_shape", "dtype_kind", "empty_view", "export_capsule", "host_view"]
