# SPDX-License-Identifier: Apache-2.0
"""Nominal Python types for opaque Aion C handles."""
from __future__ import annotations

from typing import final


class _Handle:
    raw: object
    __slots__ = ("raw",)

    def __init__(self, raw: object) -> None:
        # Public only inside the private ``aion._ffi`` package. Keeping the
        # payload on the wrapper avoids casts throughout the typed façade.
        self.raw = raw

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<opaque>)"


@final
class ContextHandle(_Handle):
    pass


@final
class TensorHandle(_Handle):
    pass


@final
class ModelHandle(_Handle):
    pass


@final
class BuilderHandle(_Handle):
    pass

__all__ = ["BuilderHandle", "ContextHandle", "ModelHandle", "TensorHandle"]
