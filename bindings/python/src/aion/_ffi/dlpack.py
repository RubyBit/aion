# SPDX-License-Identifier: Apache-2.0
"""Host memory as DLPack `DLTensor` views: the one form data crosses the C ABI in.

Anything exporting ``__dlpack__`` (numpy, PyTorch, JAX, ...) is viewed in place,
strided and in its own dtype (bfloat16 included); the core converts. Plain Python
values go through a CFFI buffer instead, so numpy stays optional.
"""
from __future__ import annotations

import ctypes
from typing import Any, Sequence

from ..enums import AionDType
from ._raw import ffi, lib

_capsule_is_valid = ctypes.pythonapi.PyCapsule_IsValid
_capsule_is_valid.restype = ctypes.c_int
_capsule_is_valid.argtypes = [ctypes.py_object, ctypes.c_char_p]
_capsule_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_capsule_pointer.restype = ctypes.c_void_p
_capsule_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_capsule_set_name = ctypes.pythonapi.PyCapsule_SetName
_capsule_set_name.restype = ctypes.c_int
_capsule_set_name.argtypes = [ctypes.py_object, ctypes.c_char_p]
# The name a consumer gives a capsule it took ownership of, so the capsule's own
# destructor no longer frees it (the consumer calls the deleter instead).
_USED = b"used_dltensor_versioned"

_READ_ONLY = 1  # DLPACK_FLAG_BITMASK_READ_ONLY
_DEVICE_NAMES = {2: "CUDA", 4: "OpenCL", 7: "Vulkan", 8: "Metal", 10: "ROCm", 15: "WebGPU"}

# (DLDataTypeCode, bits) for each scalar Aion dtype.
_DL_TYPES = {
    AionDType.AION_DTYPE_F32: (lib.kDLFloat, 32),
    AionDType.AION_DTYPE_F16: (lib.kDLFloat, 16),
    AionDType.AION_DTYPE_I32: (lib.kDLInt, 32),
    AionDType.AION_DTYPE_I8: (lib.kDLInt, 8),
}


class HostView:
    """A `DLTensor*` to hand the C ABI, and what keeps the memory it names alive."""

    __slots__ = ("ptr", "read_only", "_keep", "_capsule", "_managed")

    def __init__(
        self,
        ptr: Any,
        keep: object,
        read_only: bool = False,
        *,
        capsule: object = None,
        managed: Any = None,
    ) -> None:
        self.ptr = ptr
        self.read_only = read_only
        self._keep = keep
        self._capsule = capsule
        self._managed = managed

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(self.ptr.shape[i]) for i in range(int(self.ptr.ndim)))

    def owned(self) -> tuple[Any, bool]:
        """A `DLManagedTensorVersioned*` the core may take ownership of, and whether
        this view must still be kept alive after handing it over.

        A versioned producer's own managed tensor goes as it is: the core calls its
        deleter when done. Anything else (a legacy capsule, a CFFI buffer) is wrapped
        with no deleter, so its memory lives exactly as long as this view.
        """
        if self._managed is not None:
            return self._managed, False
        wrapper = ffi.new("DLManagedTensorVersioned*")
        wrapper.version.major = 1
        wrapper.version.minor = 3
        wrapper.deleter = ffi.NULL
        wrapper.dl_tensor = self.ptr[0]
        self._keep = (self._keep, wrapper)
        return wrapper, True

    def handed_over(self) -> None:
        """The core took ownership of `owned()`: the capsule must not free it too."""
        if self._managed is not None and _capsule_set_name(self._capsule, _USED) != 0:
            raise RuntimeError("could not mark a DLPack capsule consumed")


def view_of(obj: Any) -> HostView:
    """View `obj` through its ``__dlpack__`` export, without copying.

    Until it is handed over (`owned`/`handed_over`), the capsule stays unconsumed,
    so dropping the view lets the capsule's own destructor release the producer's
    memory, as the protocol specifies.
    """
    try:
        capsule = obj.__dlpack__(max_version=(1, 0))
    except TypeError:  # a producer from before versioned capsules
        capsule = obj.__dlpack__()
    versioned = None
    if _capsule_is_valid(capsule, b"dltensor_versioned"):
        versioned = ffi.cast("DLManagedTensorVersioned*", _capsule_pointer(capsule, b"dltensor_versioned"))
        if int(versioned.version.major) != 1:
            raise ValueError(f"DLPack major version {int(versioned.version.major)} is not supported (need 1)")
        read_only = bool(versioned.flags & _READ_ONLY)
        dl = ffi.addressof(versioned[0], "dl_tensor")
    elif _capsule_is_valid(capsule, b"dltensor"):
        legacy = ffi.cast("DLManagedTensor*", _capsule_pointer(capsule, b"dltensor"))
        read_only = False
        dl = ffi.addressof(legacy[0], "dl_tensor")
    else:
        raise TypeError(f"{type(obj).__name__}.__dlpack__ returned something that is not a DLPack capsule")
    kind = int(dl.device.device_type)
    if kind != lib.kDLCPU:
        where = _DEVICE_NAMES.get(kind, f"device type {kind}")
        raise ValueError(f"{type(obj).__name__} lives on {where}; copy it to host memory first")
    return HostView(dl, (obj, capsule), read_only, capsule=capsule, managed=versioned)


def view_of_buffer(buffer: Any, dtype: AionDType, shape: Sequence[int]) -> HostView:
    """A contiguous view of a CFFI buffer holding `shape`'s elements of `dtype`."""
    code, bits = _DL_TYPES[AionDType(int(dtype))]
    dims = ffi.new("int64_t[]", [int(d) for d in shape]) if shape else ffi.NULL
    dl = ffi.new("DLTensor*")
    dl.data = ffi.cast("void*", buffer)
    dl.device.device_type = lib.kDLCPU
    dl.ndim = len(shape)
    dl.dtype.code = code
    dl.dtype.bits = bits
    dl.dtype.lanes = 1
    dl.shape = dims
    dl.strides = ffi.NULL
    return HostView(dl, (buffer, dims))


__all__ = ["HostView", "view_of", "view_of_buffer"]
