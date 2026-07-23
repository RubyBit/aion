# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

from .enums import AionStatus


@dataclass
class AionError(RuntimeError):
    status: AionStatus
    message: str

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.status.name}: {self.message}"
