# SPDX-License-Identifier: Apache-2.0
"""Device selection helpers for the Aion Python bindings.

A tensor/model lives on exactly one device at a time. Devices are named by a
`(kind, index)` pair mirroring the Zig `DeviceSelector` (`.cpu` / `.gpu = i`).
The helpers here normalize ergonomic Python spellings (``"gpu"``, ``"gpu:1"``,
``("gpu", 1)``, ``AionDeviceKind.AION_DEVICE_GPU``) into that pair.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from .enums import AionDeviceKind, AionGpuBackend, AionGpuPower


# A device may be spelled as None (cpu), a string, a (kind,) / (kind, index)
# tuple, or an AionDeviceKind. See `_normalize_device`.
DeviceKindLike = Union[None, str, AionDeviceKind]
DeviceLike = Union[DeviceKindLike, Tuple[DeviceKindLike], Tuple[DeviceKindLike, int]]


@dataclass(frozen=True)
class GpuOptions:
    """Per-GPU registration options, mirrored to the C `AionGpuOptions` struct.

    `adapter_index=None` means auto (let the backend pick). `power`/`backend`
    accept either the `AionGpuPower`/`AionGpuBackend` enums or friendly strings
    (e.g. ``"high"``, ``"vulkan"``).
    """

    power: Union[AionGpuPower, str] = AionGpuPower.AION_GPU_POWER_DEFAULT
    backend: Union[AionGpuBackend, str] = AionGpuBackend.AION_GPU_BACKEND_ANY
    adapter_index: Optional[int] = None


_POWER_ALIASES = {
    "default": AionGpuPower.AION_GPU_POWER_DEFAULT,
    "low": AionGpuPower.AION_GPU_POWER_LOW,
    "high": AionGpuPower.AION_GPU_POWER_HIGH,
}

_BACKEND_ALIASES = {
    "any": AionGpuBackend.AION_GPU_BACKEND_ANY,
    "vulkan": AionGpuBackend.AION_GPU_BACKEND_VULKAN,
    "d3d12": AionGpuBackend.AION_GPU_BACKEND_D3D12,
    "metal": AionGpuBackend.AION_GPU_BACKEND_METAL,
    "gl": AionGpuBackend.AION_GPU_BACKEND_GL,
}


def normalize_gpu_power(power: Union[AionGpuPower, str]) -> int:
    if isinstance(power, AionGpuPower):
        return int(power)
    if isinstance(power, str):
        key = power.strip().lower()
        if key not in _POWER_ALIASES:
            raise ValueError(f"invalid gpu power: {power!r} (expected one of {sorted(_POWER_ALIASES)})")
        return int(_POWER_ALIASES[key])
    return int(power)


def normalize_gpu_backend(backend: Union[AionGpuBackend, str]) -> int:
    if isinstance(backend, AionGpuBackend):
        return int(backend)
    if isinstance(backend, str):
        key = backend.strip().lower()
        if key not in _BACKEND_ALIASES:
            raise ValueError(f"invalid gpu backend: {backend!r} (expected one of {sorted(_BACKEND_ALIASES)})")
        return int(_BACKEND_ALIASES[key])
    return int(backend)


def _normalize_device(device: DeviceLike) -> Tuple[int, int]:
    """Return ``(kind_int, index_int)`` for a device spelling.

    Accepts: ``None`` / ``"cpu"`` -> cpu; ``"gpu"`` / ``"gpu:N"`` -> gpu;
    ``("gpu", N)`` / ``(kind,)``; or an `AionDeviceKind` (index 0).
    """

    if device is None:
        return (int(AionDeviceKind.AION_DEVICE_CPU), 0)

    if isinstance(device, AionDeviceKind):
        return (int(device), 0)

    if isinstance(device, str):
        s = device.strip().lower()
        if s == "cpu":
            return (int(AionDeviceKind.AION_DEVICE_CPU), 0)
        if s == "gpu":
            return (int(AionDeviceKind.AION_DEVICE_GPU), 0)
        if s.startswith("gpu:"):
            try:
                idx = int(s[4:])
            except ValueError as e:
                raise ValueError(f"invalid device string: {device!r}") from e
            if idx < 0:
                raise ValueError("gpu index must be >= 0")
            return (int(AionDeviceKind.AION_DEVICE_GPU), idx)
        raise ValueError(f"invalid device string: {device!r} (expected 'cpu', 'gpu', or 'gpu:N')")

    if isinstance(device, (tuple, list)):
        if len(device) == 1:
            return _normalize_device(device[0])
        if len(device) == 2:
            base_kind, _ = _normalize_device(device[0])
            idx = int(device[1])
            if idx < 0:
                raise ValueError("device index must be >= 0")
            return (base_kind, idx)
        raise ValueError(f"invalid device tuple: {device!r} (expected (kind,) or (kind, index))")

    raise TypeError(f"unsupported device spec: {device!r}")


def _device_to_str(kind: int, index: int) -> str:
    if int(kind) == int(AionDeviceKind.AION_DEVICE_CPU):
        return "cpu"
    return f"gpu:{int(index)}"


def _is_cpu(device: DeviceLike) -> bool:
    kind, _ = _normalize_device(device)
    return int(kind) == int(AionDeviceKind.AION_DEVICE_CPU)


def _gpu_adapter_from_device(device: DeviceLike) -> Tuple[bool, Optional[int]]:
    """Interpret `device` for the single-GPU `load_model` convenience.

    Returns ``(is_gpu, explicit_index)``. Distinguishes ``"gpu"`` (no explicit
    physical adapter -> ``(True, None)``, let power preference pick the discrete
    GPU) from ``"gpu:N"`` / ``(kind, N)`` (explicit physical adapter -> ``(True, N)``).
    Because that convenience registers exactly one GPU, the device index names the
    physical adapter rather than a registry slot.
    """

    if device is None:
        return (False, None)
    if isinstance(device, AionDeviceKind):
        return (int(device) == int(AionDeviceKind.AION_DEVICE_GPU), None)
    if isinstance(device, str):
        s = device.strip().lower()
        if s == "cpu":
            return (False, None)
        if s == "gpu":
            return (True, None)
        if s.startswith("gpu:"):
            try:
                idx = int(s[4:])
            except ValueError as e:
                raise ValueError(f"invalid device string: {device!r}") from e
            if idx < 0:
                raise ValueError("gpu index must be >= 0")
            return (True, idx)
        raise ValueError(f"invalid device string: {device!r} (expected 'cpu', 'gpu', or 'gpu:N')")
    if isinstance(device, (tuple, list)):
        if len(device) == 1:
            return _gpu_adapter_from_device(device[0])
        if len(device) == 2:
            is_gpu, _ = _gpu_adapter_from_device(device[0])
            if not is_gpu:
                return (False, None)
            idx = int(device[1])
            if idx < 0:
                raise ValueError("device index must be >= 0")
            return (True, idx)
        raise ValueError(f"invalid device tuple: {device!r}")
    raise TypeError(f"unsupported device spec: {device!r}")
