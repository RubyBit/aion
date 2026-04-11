from __future__ import annotations

from enum import IntEnum


class AionStatus(IntEnum):
    AION_OK = 0
    AION_INVALID_ARGUMENT = 1
    AION_OUT_OF_MEMORY = 2
    AION_UNSUPPORTED = 3
    AION_INTERNAL_ERROR = 4


class AionDType(IntEnum):
    AION_DTYPE_F32 = 0
    AION_DTYPE_F16 = 1
    AION_DTYPE_I8 = 2
    AION_DTYPE_Q4_0 = 3
    AION_DTYPE_Q8_0 = 4
    AION_DTYPE_I32 = 5
