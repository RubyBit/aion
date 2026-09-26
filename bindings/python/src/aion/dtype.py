# SPDX-License-Identifier: Apache-2.0
"""One dtype vocabulary for the whole package.

`Tensor`, `Builder`, `nn`, and tracing all route dtype handling through
`normalize_dtype` and the single metadata table below. Friendly aliases (`aion.float32`, ...) are the `AionDType` enum
members re-exported under short names, so they work anywhere an `AionDType` does.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, cast

if TYPE_CHECKING:
    import numpy as np

from .enums import AionDType
from .types import DTypeLike


@dataclass(frozen=True)
class DTypeInfo:
    """What the Python layer knows about a dtype. `np_name` is None for a quantized
    dtype, which has no host element form (no numpy array, no DLPack export)."""

    enum: AionDType
    np_name: Optional[str]   # numpy dtype name, or None for quantized
    is_quantized: bool


_TABLE: dict[AionDType, DTypeInfo] = {
    AionDType.AION_DTYPE_F32:  DTypeInfo(AionDType.AION_DTYPE_F32,  "float32", False),
    AionDType.AION_DTYPE_F16:  DTypeInfo(AionDType.AION_DTYPE_F16,  "float16", False),
    AionDType.AION_DTYPE_I8:   DTypeInfo(AionDType.AION_DTYPE_I8,   "int8",    False),
    AionDType.AION_DTYPE_I32:  DTypeInfo(AionDType.AION_DTYPE_I32,  "int32",   False),
    AionDType.AION_DTYPE_Q8_0: DTypeInfo(AionDType.AION_DTYPE_Q8_0, None,      True),
    AionDType.AION_DTYPE_Q4_0: DTypeInfo(AionDType.AION_DTYPE_Q4_0, None,      True),
}

_DISPLAY_NAMES: dict[AionDType, str] = {
    AionDType.AION_DTYPE_F32: "float32",
    AionDType.AION_DTYPE_F16: "float16",
    AionDType.AION_DTYPE_I8: "int8",
    AionDType.AION_DTYPE_I32: "int32",
    AionDType.AION_DTYPE_Q8_0: "q8_0",
    AionDType.AION_DTYPE_Q4_0: "q4_0",
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


def normalize_dtype(x: DTypeLike) -> AionDType:
    """Map an Aion constant or supported NumPy dtype to ``AionDType``."""
    if isinstance(x, AionDType):
        return x
    if isinstance(x, str):
        raise TypeError(
            f"dtype strings are not supported; use an Aion constant such as "
            f"aion.float32 instead of {x!r}"
        )
    if isinstance(x, int):
        raise TypeError(
            "raw dtype ordinals are not supported; use an Aion dtype constant "
            "such as aion.float32"
        )
    name = _np_name(x)
    if name is not None:
        try:
            return _NP_TO_ENUM[name]
        except KeyError:
            raise ValueError(f"no Aion dtype for numpy dtype {name!r}") from None
    raise TypeError(
        f"dtype must be an Aion dtype constant or NumPy dtype, got "
        f"{type(x).__name__}"
    )


def dtype_name(dtype: DTypeLike) -> str:
    """Return the concise public name for an Aion dtype."""
    return _DISPLAY_NAMES[normalize_dtype(dtype)]


def info(dtype: DTypeLike) -> DTypeInfo:
    """Metadata record for a dtype (accepts any `normalize_dtype` input)."""
    return _TABLE[normalize_dtype(dtype)]


def numpy_dtype(dtype: DTypeLike) -> "np.dtype[Any]":
    """numpy dtype for a scalar Aion dtype; raises for quantized dtypes."""
    di = info(dtype)
    if di.np_name is None:
        raise NotImplementedError(
            f"{dtype_name(di.enum)} has no host numpy representation"
        )
    import numpy as np

    return np.dtype(di.np_name)


def is_quantized(dtype: DTypeLike) -> bool:
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
    "dtype_name",
    "normalize_dtype",
    "info",
    "numpy_dtype",
    "is_quantized",
    "float32",
    "float16",
    "int8",
    "int32",
    "q8_0",
    "q4_0",
]
