# SPDX-License-Identifier: Apache-2.0
"""One dtype vocabulary for the whole package.

`Tensor`, `Builder`, `nn`, and tracing all route dtype handling through
`normalize_dtype`, and read/write buffers are sized from the single metadata
table below. Friendly aliases (`aion.float32`, ...) are the `AionDType` enum
members re-exported under short names, so they work anywhere an `AionDType` does.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, cast

if TYPE_CHECKING:
    import numpy as np

from .enums import AionDType


@dataclass(frozen=True)
class DTypeInfo:
    """Everything the Python layer needs to move a dtype across the C ABI.

    `np_name`/`c_elem` are None for quantized dtypes (no host scalar form):
    those have no numpy array representation and no per-element read/write path.
    """

    enum: AionDType
    np_name: Optional[str]   # numpy dtype name, or None for quantized
    c_elem: Optional[str]    # cffi element type for ffi.new / ffi.from_buffer
    itemsize: int            # bytes per logical element (0 for quantized)
    is_quantized: bool


# cffi has no half type, so f16's buffer element is uint16_t: the C ABI's
# read/write take a void* + element *count*, and 2 bytes/elem over n elems is a
# byte-exact reinterpret of an np.float16 array (same trick Tensor.zero uses).
_TABLE: dict[AionDType, DTypeInfo] = {
    AionDType.AION_DTYPE_F32:  DTypeInfo(AionDType.AION_DTYPE_F32,  "float32", "float",    4, False),
    AionDType.AION_DTYPE_F16:  DTypeInfo(AionDType.AION_DTYPE_F16,  "float16", "uint16_t", 2, False),
    AionDType.AION_DTYPE_I8:   DTypeInfo(AionDType.AION_DTYPE_I8,   "int8",    "int8_t",   1, False),
    AionDType.AION_DTYPE_I32:  DTypeInfo(AionDType.AION_DTYPE_I32,  "int32",   "int32_t",  4, False),
    AionDType.AION_DTYPE_Q8_0: DTypeInfo(AionDType.AION_DTYPE_Q8_0, None,      None,       0, True),
    AionDType.AION_DTYPE_Q4_0: DTypeInfo(AionDType.AION_DTYPE_Q4_0, None,      None,       0, True),
}

# String spellings accepted everywhere a dtype is expected.
_ALIASES: dict[str, AionDType] = {
    "f32": AionDType.AION_DTYPE_F32,
    "float32": AionDType.AION_DTYPE_F32,
    "f16": AionDType.AION_DTYPE_F16,
    "float16": AionDType.AION_DTYPE_F16,
    "half": AionDType.AION_DTYPE_F16,
    "i8": AionDType.AION_DTYPE_I8,
    "int8": AionDType.AION_DTYPE_I8,
    "i32": AionDType.AION_DTYPE_I32,
    "int32": AionDType.AION_DTYPE_I32,
    "q8_0": AionDType.AION_DTYPE_Q8_0,
    "q4_0": AionDType.AION_DTYPE_Q4_0,
}

# numpy dtype name -> enum, derived from the table (scalar dtypes only).
_NP_TO_ENUM: dict[str, AionDType] = {
    info.np_name: info.enum for info in _TABLE.values() if info.np_name is not None
}


def _np_name(x: object) -> Optional[str]:
    """`np.dtype(x).name` if numpy is present and `x` names a dtype, else None."""
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        return np.dtype(cast(Any, x)).name
    except TypeError:
        return None


def normalize_dtype(x: object) -> AionDType:
    """Map an `AionDType` | dtype string | numpy dtype | int to `AionDType`."""
    if isinstance(x, AionDType):
        return x
    if isinstance(x, str):
        key = x.lower()
        try:
            return _ALIASES[key]
        except KeyError:
            raise ValueError(f"unknown dtype {x!r}") from None
    # bool is an int subclass; treat it as a numpy/other dtype, not enum ordinal.
    if isinstance(x, int) and not isinstance(x, bool):
        return AionDType(int(x))
    name = _np_name(x)
    if name is not None:
        try:
            return _NP_TO_ENUM[name]
        except KeyError:
            raise ValueError(f"no Aion dtype for numpy dtype {name!r}") from None
    raise ValueError(f"cannot interpret {x!r} as a dtype")


def info(dtype: object) -> DTypeInfo:
    """Metadata record for a dtype (accepts any `normalize_dtype` input)."""
    return _TABLE[normalize_dtype(dtype)]


def numpy_dtype(dtype: object) -> "np.dtype[Any]":
    """numpy dtype for a scalar Aion dtype; raises for quantized dtypes."""
    di = info(dtype)
    if di.np_name is None:
        raise NotImplementedError(f"{di.enum.name} has no host numpy representation")
    import numpy as np

    return np.dtype(di.np_name)


def c_elem(dtype: object) -> str:
    """cffi element type name for a scalar Aion dtype; raises for quantized."""
    di = info(dtype)
    if di.c_elem is None:
        raise NotImplementedError(f"{di.enum.name} has no per-element host buffer type")
    return di.c_elem


def is_quantized(dtype: object) -> bool:
    return info(dtype).is_quantized


# Friendly public aliases (enum members under short names).
float32 = AionDType.AION_DTYPE_F32
float16 = AionDType.AION_DTYPE_F16
int8 = AionDType.AION_DTYPE_I8
int32 = AionDType.AION_DTYPE_I32
q8_0 = AionDType.AION_DTYPE_Q8_0
q4_0 = AionDType.AION_DTYPE_Q4_0

__all__ = [
    "DTypeInfo",
    "normalize_dtype",
    "info",
    "numpy_dtype",
    "c_elem",
    "is_quantized",
    "float32",
    "float16",
    "int8",
    "int32",
    "q8_0",
    "q4_0",
]
