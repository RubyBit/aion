# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import atexit
import os
import weakref
from typing import Optional, Sequence

from .device import GpuOptions, build_gpu_options_array
from .errors import raise_for_status
from ._ffi import ffi, lib

_DEFAULT_CONTEXT = None


def _default_thread_count_from_env() -> Optional[int]:
    """Resolve thread count for the implicit default context from env vars.

    Precedence:
      1. AION_DEFAULT_THREAD_COUNT
      2. AION_THREAD_COUNT
    """

    raw = os.getenv("AION_DEFAULT_THREAD_COUNT")
    if raw is None:
        raw = os.getenv("AION_THREAD_COUNT")
    if raw is None:
        return None

    raw = raw.strip()
    if raw == "":
        raise ValueError("AION_DEFAULT_THREAD_COUNT/AION_THREAD_COUNT must not be empty")

    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(
            f"AION_DEFAULT_THREAD_COUNT/AION_THREAD_COUNT must be an integer, got: {raw!r}"
        ) from e

    if value <= 0:
        raise ValueError("AION_DEFAULT_THREAD_COUNT/AION_THREAD_COUNT must be > 0")
    return value


class Context:
    """Owns an `AionContext*`.

    Notes:
            - Tensor/model handles are separate objects; Aion requires destroying them
                before destroying the context. `close()` handles this automatically by
                closing any still-live children first.
    """

    def __init__(
        self,
        thread_count: Optional[int] = None,
        *,
        gpus: Optional[Sequence[GpuOptions]] = None,
    ):
        """Create a context.

        Registering GPUs via `gpus` makes them selectable as `.gpu = i`
        (in Python: ``"gpu:i"``) for `tensor.to(...)` and `LoadedModel.load(device=)`.
        `gpus[i]` becomes device index `i`. On a runtime built without GPU support
        (`-Dgpu`), a non-empty `gpus` raises `AionError` (`AION_UNSUPPORTED`).
        """

        if thread_count is None:
            thread_count = os.cpu_count() or 1
        thread_count = int(thread_count)
        if thread_count <= 0:
            raise ValueError("thread_count must be > 0")

        gpu_list = list(gpus) if gpus else []

        out_ctx = ffi.new("AionContext**")
        gpus_arr, gpu_count = build_gpu_options_array(ffi, gpu_list)
        st = lib.aion_context_create(thread_count, gpus_arr, gpu_count, out_ctx)
        raise_for_status(st, None, what="aion_context_create")

        self._ctx = out_ctx[0]
        self._closed = False
        self._children: weakref.WeakSet[object] = weakref.WeakSet()

    @classmethod
    def gpu(
        cls,
        *,
        thread_count: Optional[int] = None,
        power="high",
        backend="any",
        adapter_index: Optional[int] = None,
    ) -> "Context":
        """Convenience: create a context with a single GPU registered as ``gpu:0``."""

        return cls(
            thread_count=thread_count,
            gpus=[GpuOptions(power=power, backend=backend, adapter_index=adapter_index)],
        )

    @property
    def ptr(self):
        return self._ctx

    def _register_child(self, obj: object) -> None:
        self._children.add(obj)

    def _unregister_child(self, obj: object) -> None:
        # WeakSet supports discard (no KeyError if missing).
        self._children.discard(obj)

    def close(self) -> None:
        if self._closed:
            return

        # Best-effort deterministic cleanup of live children before context destroy.
        for child in list(self._children):
            try:
                close_fn = getattr(child, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                # Keep cleanup resilient: try closing remaining children.
                pass

        lib.aion_context_destroy(self._ctx)
        self._ctx = ffi.NULL
        self._closed = True

    def __enter__(self) -> "Context":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):  # pragma: no cover
        try:
            if not getattr(self, "_closed", True):
                # Best-effort cleanup: do not raise from finalizer.
                self.close()
        except Exception:
            pass

    # Convenience factories.
    def tensor_empty(self, shape: Sequence[int], *, dtype=None):
        from .tensor import Tensor

        return Tensor.empty(self, shape, dtype=dtype)

    def tensor_from_f32(self, shape: Sequence[int], values):
        from .tensor import Tensor

        return Tensor.from_f32(self, shape, values)

    def scalar_f32(self, value: float):
        from .tensor import Tensor

        return Tensor.scalar_f32(self, value)

    def load_model(self, path: str):
        from .model import LoadedModel

        return LoadedModel.load(self, path)


def get_default_context() -> Context:
    """Process-wide default context for convenience constructors.

    Thread count can be configured via:
      - AION_DEFAULT_THREAD_COUNT
      - AION_THREAD_COUNT
    """

    global _DEFAULT_CONTEXT
    if _DEFAULT_CONTEXT is None or getattr(_DEFAULT_CONTEXT, "_closed", True):
        tc = _default_thread_count_from_env()
        _DEFAULT_CONTEXT = Context(thread_count=tc)
    return _DEFAULT_CONTEXT


def reset_default_context() -> None:
    """Reset/close the process-wide default context.

    Useful when env-based configuration changes mid-process (e.g. tests).
    """

    global _DEFAULT_CONTEXT
    if _DEFAULT_CONTEXT is not None and not getattr(_DEFAULT_CONTEXT, "_closed", True):
        _DEFAULT_CONTEXT.close()
    _DEFAULT_CONTEXT = None


@atexit.register
def _close_default_context_atexit() -> None:
    """Tear down the default context deterministically at interpreter exit.

    atexit runs while the interpreter is still healthy, so `Context.close()`
    frees all children (tensors/builders/models) in a controlled order before
    the C ABI/cffi module is finalized — avoiding the arbitrary-order handle
    teardown that can crash a not-explicitly-closed session at shutdown.
    """
    try:
        reset_default_context()
    except Exception:
        pass
