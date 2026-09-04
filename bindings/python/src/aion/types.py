# SPDX-License-Identifier: Apache-2.0
"""Shared type aliases for the public API.

numpy is an optional dependency, so its precise types are only used at
type-check time (`TYPE_CHECKING`); at runtime the aliases fall back to `Any` so
the package imports without numpy installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Sequence, TypeAlias

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
    # Aion constants plus NumPy's dtype classes/instances. Dtype strings and raw
    # enum ordinals are deliberately not part of the pre-release public API.
    DTypeLike: TypeAlias = AionDType | np.dtype[Any] | type[np.generic]
else:  # runtime: no hard numpy dependency
    ArrayLike = Any
    F32ArrayLike = Any
    NDArray = Any
    NDArrayF32 = Any
    DTypeLike: TypeAlias = AionDType

# A tensor shape (all concrete ints).
Shape: TypeAlias = Sequence[int]


@dataclass(frozen=True)
class AttentionWindow:
    """Keys a query may attend to: `[anchor - left, anchor + span + right)`, where
    `anchor` is the query (its chunk start when `chunk` is set) and `span` is
    `chunk` or 1. Mirrors `graph.AttentionWindow`."""

    UNBOUNDED: ClassVar[int] = 0xFFFFFFFF
    FULL: ClassVar["AttentionWindow"]
    CAUSAL: ClassVar["AttentionWindow"]

    left: int = UNBOUNDED
    right: int = UNBOUNDED
    chunk: int = 0

    @classmethod
    def sliding(cls, left: int, right: int = 0) -> "AttentionWindow":
        """`left` keys before the query and `right` after it, the query included."""
        return cls(left=left, right=right)

    @classmethod
    def chunked(cls, size: int, left: int = 0) -> "AttentionWindow":
        """Every query in a chunk of `size` sees that chunk plus `left` keys before it."""
        return cls(left=left, right=0, chunk=size)


AttentionWindow.FULL = AttentionWindow()
AttentionWindow.CAUSAL = AttentionWindow(right=0)

# f32 scalar / value conveniences.
F32Scalar: TypeAlias = float
F32Values: TypeAlias = Iterable[float]

__all__ = [
    "ArrayLike",
    "AttentionWindow",
    "DTypeLike",
    "F32ArrayLike",
    "F32Scalar",
    "F32Values",
    "NDArray",
    "NDArrayF32",
    "Shape",
]
