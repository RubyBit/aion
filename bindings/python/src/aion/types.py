# SPDX-License-Identifier: Apache-2.0
"""Shared type aliases for the public API.

numpy is an optional dependency, so its precise types are only used at
type-check time (`TYPE_CHECKING`); at runtime the aliases fall back to `Any` so
the package imports without numpy installed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Sequence, TypeAlias, Union

from .enums import AionDType

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    # Anything numpy can turn into an array: ndarray, nested sequences, scalars.
    ArrayLike: TypeAlias = npt.ArrayLike
    F32ArrayLike: TypeAlias = npt.ArrayLike
    # Concrete array results.
    NDArray: TypeAlias = "npt.NDArray[Any]"
    NDArrayF32: TypeAlias = "npt.NDArray[np.float32]"
else:  # runtime: no hard numpy dependency
    ArrayLike = Any
    F32ArrayLike = Any
    NDArray = Any
    NDArrayF32 = Any

# A tensor shape (all concrete ints).
Shape: TypeAlias = Sequence[int]

# Anything accepted where a dtype is expected. Kept here rather than spelled out
# per call site so `dtype=` means one thing across the whole API; `normalize_dtype`
# is what turns it into an `AionDType`.
DTypeLike: TypeAlias = Union[str, "AionDType"]

# f32 scalar / value conveniences.
F32Scalar: TypeAlias = float
F32Values: TypeAlias = Iterable[float]

__all__ = [
    "ArrayLike",
    "DTypeLike",
    "F32ArrayLike",
    "F32Scalar",
    "F32Values",
    "NDArray",
    "NDArrayF32",
    "Shape",
]
