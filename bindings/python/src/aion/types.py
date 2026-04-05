from __future__ import annotations

from typing import Any, Iterable, Sequence, TypeAlias


Shape: TypeAlias = Sequence[int]

# f32-focused for the current C ABI.
F32Scalar: TypeAlias = float
F32Values: TypeAlias = Iterable[float]

# Keep API typing lightweight and runtime-safe.
NDArrayF32: TypeAlias = Any
ArrayLike: TypeAlias = Any
F32ArrayLike: TypeAlias = Any
