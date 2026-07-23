# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Iterable, Mapping, Optional, overload

from .device import DeviceLike, _is_cpu, _normalize_device
from ._ffi.handles import ModelHandle
from ._ffi.runtime import (
    bind_model_input,
    destroy_model,
    load_model,
    model_count,
    model_dtype,
    model_name,
    model_output_is_state,
    model_output_tensor,
    model_position,
    model_rank,
    reset_model_state,
    run_model,
    set_model_position,
    set_state_input_policy,
)
from .enums import AionDType
from .types import NDArray

if TYPE_CHECKING:
    from .context import Context
    from .tensor import Tensor

# What model I/O accepts: a numpy array (or array-like) or an existing Tensor.
InputValue = object


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: AionDType
    rank: int


class LoadedModel:
    """Owns an `AionLoadedModel*` handle; keeps the owning Context alive."""

    def __init__(self, ctx: "Context", ptr: ModelHandle) -> None:
        # Strong ref: model must keep its Context alive.
        self._ctx_owner = ctx
        self._ctx_owner._register_child(self)
        self._m: ModelHandle | None = ptr
        self._closed = False
        # A model produced by tracing/Builder.compile shares (does not copy) the
        # builder's weight tensors, so the builder must outlive the model. When
        # set, it is closed together with the model.
        self._authoring_builder = None

    def _attach_authoring_builder(self, builder: object) -> None:
        # A traced/compiled model shares the builder's weight tensor *handles*, but
        # the underlying storage is Context-owned and freed only at context
        # destroy, so child teardown order is safe. The builder stays a context
        # child (torn down in the context's controlled sequence like any other);
        # this strong ref lets an explicit model.close() also close it promptly.
        self._authoring_builder = builder

    @property
    def ptr(self) -> ModelHandle:
        return self._require_handle()

    def _require_handle(self) -> ModelHandle:
        if self._m is None:
            raise RuntimeError("LoadedModel is closed")
        return self._m

    @property
    def context(self) -> "Context":
        """Owning context for this loaded model."""

        return self._ctx_owner

    def close(self) -> None:
        if self._closed:
            return
        handle = self._m
        if handle is not None:
            destroy_model(handle)
        try:
            self._ctx_owner._unregister_child(self)
        except Exception:
            pass
        self._m = None
        self._closed = True
        # Drop the strong ref to the traced builder. Do NOT close it here: it is a
        # context child torn down in the context's controlled sequence, and closing
        # it mid-teardown races the context finalizer. Dropping the ref lets it be
        # collected once nothing else references it.
        self._authoring_builder = None

    def __enter__(self) -> "LoadedModel":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover
        # Best-effort only; the attached builder (if any) is a context child and
        # is torn down in the context's controlled sequence — don't close it here
        # (finalizer order vs the context is unspecified).
        try:
            if not getattr(self, "_closed", True):
                handle = getattr(self, "_m", None)
                if handle is not None:
                    destroy_model(handle)
        except Exception:
            pass

    @classmethod
    def load(
        cls,
        ctx: "Context",
        path: str,
        *,
        device: DeviceLike = None,
        cache_capacity: int | None = None,
        growable: bool = False,
        initial_cache_capacity: int = 8,
        auto_positions: bool = True,
    ) -> "LoadedModel":
        """Load a model onto `device` (default CPU).

        The GPU it targets must be registered on `ctx` (see `Context(gpus=...)`
        or `Context.gpu()`). The model's backend and tiling follow the device;
        keep binding CPU input tensors — the runtime migrates them on `run()` and
        flushes outputs back to host for reading. Do NOT bind a device-resident
        tensor (from `tensor.to("gpu")`) as a model input.

        Role-declared state (models converted with input roles):
        - `cache_capacity`: capacity in tokens for every sequence-cache input
          whose capacity axis is a free dim symbol; the runtime allocates and
          zeroes those caches itself (no `bind_input` needed).
        - `growable=True`: start those caches at `initial_cache_capacity` tokens
          and grow them on demand up to `cache_capacity`.
        - `auto_positions`: feed role-declared `cache_write_index` /
          `cache_visible_end` / `positions` inputs from the runtime-tracked
          position (advanced per run by the tokens input's sequence length).
          Binding one of those inputs manually always overrides its auto value.
        """

        if not isinstance(path, str):
            raise TypeError("path must be a str")

        absolute = os.path.isabs(path)
        device_kind: int | None = None
        device_index = 0
        if device is not None and not _is_cpu(device):
            device_kind, device_index = _normalize_device(device)

        handle = load_model(
            ctx.ptr,
            path,
            absolute=absolute,
            device_kind=device_kind,
            device_index=device_index,
            cache_capacity=cache_capacity,
            growable=growable,
            initial_cache_capacity=initial_cache_capacity,
            auto_positions=auto_positions,
        )
        return cls(ctx, handle)

    def input_count(self) -> int:
        return model_count(self._require_handle(), "input")

    def output_count(self) -> int:
        return model_count(self._require_handle(), "output")

    def input_names(self) -> list[str]:
        n = self.input_count()
        return [model_name(self._ctx_owner.ptr, self._require_handle(), "input", i) for i in range(n)]

    def output_names(self) -> list[str]:
        n = self.output_count()
        return [model_name(self._ctx_owner.ptr, self._require_handle(), "output", i) for i in range(n)]

    def input_specs(self) -> list[TensorSpec]:
        names = self.input_names()
        out: list[TensorSpec] = []
        for i, name in enumerate(names):
            out.append(TensorSpec(name=name, dtype=self.input_dtype(i), rank=self.input_rank(i)))
        return out

    def output_specs(self) -> list[TensorSpec]:
        names = self.output_names()
        out: list[TensorSpec] = []
        for i, name in enumerate(names):
            out.append(TensorSpec(name=name, dtype=self.output_dtype(i), rank=self.output_rank(i)))
        return out

    def input_dtype(self, index: int) -> AionDType:
        return model_dtype(self._ctx_owner.ptr, self._require_handle(), "input", index)

    def input_rank(self, index: int) -> int:
        return model_rank(self._ctx_owner.ptr, self._require_handle(), "input", index)

    def output_dtype(self, index: int) -> AionDType:
        return model_dtype(self._ctx_owner.ptr, self._require_handle(), "output", index)

    def output_rank(self, index: int) -> int:
        return model_rank(self._ctx_owner.ptr, self._require_handle(), "output", index)

    def bind_input(self, name: str, tensor: "Tensor") -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a str")
        bind_model_input(
            self._ctx_owner.ptr, self._require_handle(), name, tensor.ptr
        )

    def set_state_input_growable(
        self,
        name: str,
        *,
        initial_capacity: int,
        max_capacity: int,
        growth_numerator: int = 2,
        growth_denominator: int = 1,
    ) -> None:
        """Make an io-aliased recurrent-state input grow on demand.

        The state slot starts at `initial_capacity` tokens and the runtime grows
        it (device-resident growth included) up to `max_capacity` as writes cross
        the current size — so callers need not pre-allocate the maximum. `growth_*`
        is the geometric factor (default 2x).
        """
        if not isinstance(name, str):
            raise TypeError("name must be a str")
        set_state_input_policy(
            self._ctx_owner.ptr,
            self._require_handle(),
            name,
            kind="growable",
            initial_capacity=initial_capacity,
            max_capacity=max_capacity,
            growth_numerator=growth_numerator,
            growth_denominator=growth_denominator,
        )

    def set_state_input_ring(self, name: str, *, window: int) -> None:
        """Make an io-aliased recurrent-state input a fixed ring buffer of `window` tokens."""
        if not isinstance(name, str):
            raise TypeError("name must be a str")
        set_state_input_policy(
            self._ctx_owner.ptr,
            self._require_handle(),
            name,
            kind="ring",
            window=window,
        )

    def output_is_state(self, index: int) -> bool:
        """Whether output `index` is io-aliased recurrent state (a `next_*` carry
        the runtime already writes back into its input slot every run)."""
        return model_output_is_state(
            self._ctx_owner.ptr, self._require_handle(), index
        )

    @property
    def position(self) -> int:
        """Tokens consumed so far by position auto-management (0 when disabled)."""
        return model_position(self._ctx_owner.ptr, self._require_handle())

    def set_position(self, tokens: int) -> None:
        """Overwrite the auto-tracked position (session restore / rollback)."""
        set_model_position(self._ctx_owner.ptr, self._require_handle(), tokens)

    def _default_output_names(self) -> list[str]:
        """All outputs except io-aliased state carries (KV caches etc.), which the
        runtime already persists — copying them out per run is pure overhead.
        Falls back to all outputs when every output is a state carry."""
        cached = getattr(self, "_default_outputs", None)
        if cached is not None:
            return cached
        names = self.output_names()
        non_state = [n for i, n in enumerate(names) if not self.output_is_state(i)]
        result = non_state if non_state else names
        self._default_outputs = result
        return result

    def run_tensors(
        self,
        inputs: Mapping[str, "Tensor"],
        *,
        outputs: Iterable[str] | None = None,
    ) -> "dict[str, Tensor]":
        """Run the model and return output tensors.

        By default only non-state outputs are returned (io-aliased carries like
        `next_k_cache.*` are skipped — the runtime persists them itself); pass
        `outputs=[...]` to select any output explicitly.
        Caller owns the returned output tensor *handles* and must `close()` them.
        """

        for name, t in inputs.items():
            self.bind_input(name, t)

        self.run()

        out_names = list(outputs) if outputs is not None else self._default_output_names()
        return {name: self.output_tensor(name) for name in out_names}

    def run_numpy(
        self,
        inputs: Mapping[str, InputValue],
        *,
        outputs: Iterable[str] | None = None,
    ) -> dict[str, NDArray]:
        """Run the model with Tensor or NumPy inputs and return NumPy outputs.

        By default only non-state outputs are returned (io-aliased carries like
        `next_k_cache.*` are skipped — the runtime persists them itself); pass
        `outputs=[...]` to select any output explicitly.
        Output tensor handles are closed internally; returned arrays own their data.
        """

        from .tensor import Tensor

        temps: list[Tensor] = []
        try:
            for name, v in inputs.items():
                if isinstance(v, Tensor):
                    t = v
                else:
                    t = Tensor(v, ctx=self._ctx_owner)
                    temps.append(t)
                self.bind_input(name, t)

            self.run()

            out_names = list(outputs) if outputs is not None else self._default_output_names()
            outs: dict[str, NDArray] = {}
            for name in out_names:
                t_out = self.output_tensor(name)
                try:
                    outs[name] = t_out.numpy()
                finally:
                    t_out.close()
            return outs
        finally:
            for t in temps:
                try:
                    t.close()
                except Exception:
                    pass

    @overload
    def run(self) -> None: ...
    @overload
    def run(self, inputs: Mapping[str, InputValue], *, outputs: Iterable[str] | None = None) -> dict[str, NDArray]: ...

    def run(
        self,
        inputs: Optional[Mapping[str, InputValue]] = None,
        *,
        outputs: Iterable[str] | None = None,
    ) -> Optional[dict[str, NDArray]]:
        """Run the model.

        - `run()` (no args): execute using already-bound input tensors and return
          `None` (the low-level fast path).
        - `run({"x": arr})`: bind numpy/Tensor inputs, execute, and return a
          `{name: numpy}` dict (sugar over `run_numpy`).
        """
        if inputs is None:
            run_model(self._ctx_owner.ptr, self._require_handle())
            return None
        return self.run_numpy(inputs, outputs=outputs)

    def reset_state(self) -> None:
        """Zero all recurrent (io-aliased) input state — KV caches, LSTM h/c, etc.

        Call this between independent sequences (e.g. utterances) instead of
        reloading the model. No-op before the first ``run()``.
        """
        reset_model_state(self._ctx_owner.ptr, self._require_handle())

    def output_tensor(self, name: str) -> "Tensor":
        if not isinstance(name, str):
            raise TypeError("name must be a str")
        from .tensor import Tensor

        handle = model_output_tensor(
            self._ctx_owner.ptr, self._require_handle(), name
        )
        return Tensor._from_handle(self._ctx_owner, handle)
