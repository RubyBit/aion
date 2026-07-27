# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import atexit
import os
from types import TracebackType
import weakref
from typing import TYPE_CHECKING, Optional, Sequence

from .device import GpuOptions
from ._ffi.handles import ContextHandle
from ._ffi.runtime import create_context, destroy_context

if TYPE_CHECKING:
    from .builder import Builder
    from .model import LoadedModel
    from .tensor import Tensor

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
    ) -> None:
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

        self._ctx: ContextHandle | None = create_context(thread_count, gpu_list)
        self._closed = False
        self._children: weakref.WeakSet[object] = weakref.WeakSet()
        # Graph owner for values built without an explicit Builder (see
        # `scratch_builder`). Strong ref: nothing else keeps it alive.
        self._scratch: "Builder | None" = None

    @classmethod
    def gpu(
        cls,
        *,
        thread_count: Optional[int] = None,
        power: str = "high",
        backend: str = "any",
        adapter_index: Optional[int] = None,
    ) -> "Context":
        """Convenience: create a context with a single GPU registered as ``gpu:0``."""

        return cls(
            thread_count=thread_count,
            gpus=[GpuOptions(power=power, backend=backend, adapter_index=adapter_index)],
        )

    @property
    def ptr(self) -> ContextHandle:
        if self._ctx is None:
            raise RuntimeError("Context is closed")
        return self._ctx

    def scratch_builder(self) -> "Builder":
        """A builder for values created without naming one — lazily made, then reused.

        `aion.tensor(data)` produces *data*, and composing ops needs a graph, so
        something has to own one. This is that owner, so exploratory work costs no
        ceremony:

            x = aion.tensor(np_x, ctx=ctx)
            print((x @ w).relu())          # no `with aion.Builder(...)` needed

        Safe to share because `compile` prunes to the requested outputs: values you
        explored and did not ask for never reach a compiled model. It does grow
        monotonically across a long session, though — a fresh `Context` clears it.

        Models should still use an explicit `Builder`: one graph per model keeps
        ownership and teardown obvious.
        """
        from .builder import Builder

        if self._scratch is None or self._scratch._closed:
            self._scratch = Builder(self)
        return self._scratch

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

        self._scratch = None

        handle = self._ctx
        if handle is not None:
            destroy_context(handle)
        self._ctx = None
        self._closed = True

    def __enter__(self) -> "Context":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover
        try:
            if not getattr(self, "_closed", True):
                # Best-effort cleanup: do not raise from finalizer.
                self.close()
        except Exception:
            pass

    def load_model(self, path: str) -> "LoadedModel":
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
