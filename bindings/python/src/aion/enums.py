# SPDX-License-Identifier: Apache-2.0
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


class AionDeviceKind(IntEnum):
    AION_DEVICE_CPU = 0
    AION_DEVICE_GPU = 1


class AionGpuPower(IntEnum):
    AION_GPU_POWER_DEFAULT = 0
    AION_GPU_POWER_LOW = 1
    AION_GPU_POWER_HIGH = 2


class AionGpuBackend(IntEnum):
    AION_GPU_BACKEND_ANY = 0
    AION_GPU_BACKEND_VULKAN = 1
    AION_GPU_BACKEND_D3D12 = 2
    AION_GPU_BACKEND_METAL = 3
    AION_GPU_BACKEND_GL = 4


# --- Model authoring (Builder) enums -----------------------------------------


class AionUnaryOp(IntEnum):
    AION_UNARY_RELU = 0
    AION_UNARY_GELU = 1
    AION_UNARY_SILU = 2
    AION_UNARY_SIGMOID = 3
    AION_UNARY_TANH = 4
    AION_UNARY_SQRT = 5
    AION_UNARY_LOG = 6


class AionBinaryOp(IntEnum):
    AION_BINARY_ADD = 0
    AION_BINARY_SUB = 1
    AION_BINARY_MUL = 2
    AION_BINARY_DIV = 3
    AION_BINARY_EQ = 4
    AION_BINARY_NE = 5
    AION_BINARY_LT = 6
    AION_BINARY_GT = 7
    AION_BINARY_LE = 8
    AION_BINARY_GE = 9
    # Gated activation: act(a) * b, with the activation in ElemwiseAttrs.act.
    AION_BINARY_GATE = 10


class AionReduceOp(IntEnum):
    AION_REDUCE_SUM = 0
    AION_REDUCE_MEAN = 1


class AionPadMode(IntEnum):
    AION_PAD_ZERO = 0
    AION_PAD_REFLECT = 1


class AionInputRoleKind(IntEnum):
    AION_ROLE_SEQUENCE_CACHE = 1
    AION_ROLE_CACHE_WRITE_INDEX = 2
    AION_ROLE_CACHE_VISIBLE_END = 3
    AION_ROLE_POSITIONS = 4
    AION_ROLE_TOKENS = 5
    AION_ROLE_STATE = 6


class AionOp(IntEnum):
    AION_OP_MATMUL = 0
    AION_OP_MATMUL_NT = 1
    AION_OP_ELEMWISE = 2
    AION_OP_UNARY = 3
    AION_OP_SOFTMAX = 4
    AION_OP_LAYERNORM = 5
    AION_OP_RMSNORM = 6
    AION_OP_ATTENTION = 7
    AION_OP_RELPOS_MHA = 8
    AION_OP_CONV1D = 9
    AION_OP_CONV2D = 10
    AION_OP_COPY = 11
    AION_OP_ROPE1D = 12
    AION_OP_SEQUENCE_APPEND = 13
    AION_OP_REDUCE = 14
    AION_OP_CONCAT = 15
    AION_OP_RESHAPE = 16
    AION_OP_SQUEEZE = 17
    AION_OP_UNSQUEEZE = 18
    AION_OP_TRANSPOSE2D = 19
    AION_OP_SLICE = 20
    AION_OP_LSTM_CELL = 21
    AION_OP_RFFT = 22
    AION_OP_STFT = 23
    AION_OP_CAST = 24
    AION_OP_ARGMAX = 25
    AION_OP_SCATTER_ROW = 26
    # 27 was AION_OP_GELU_MUL, retired: a gated activation is AION_OP_ELEMWISE with
    # op=AION_BINARY_GATE and act naming the activation.
    AION_OP_GATHER = 28
    AION_OP_DIM = 29
    AION_OP_IOTA = 30
