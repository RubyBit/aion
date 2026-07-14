# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping

from .device import DeviceLike, _is_cpu, _normalize_device
from .errors import raise_for_status
from ._ffi import ffi, lib
from .enums import AionDType
from .types import ArrayLike, NDArrayF32


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: AionDType
    rank: int


def _get_indexed_name(ctx_ptr, fn, m_ptr, index: int) -> str:
    out_len = ffi.new("size_t*")
    st = fn(m_ptr, index, ffi.NULL, 0, out_len)
    # Even if this fails, propagate via raise_for_status for a useful message.
    raise_for_status(st, ctx_ptr, what="get_name")

    n = int(out_len[0])
    if n <= 0:
        return ""
    buf = ffi.new("char[]", n + 1)
    st = fn(m_ptr, index, buf, n + 1, out_len)
    raise_for_status(st, ctx_ptr, what="get_name")
    raw: bytes = ffi.string(buf)
    return raw.decode("utf-8", errors="replace")


class LoadedModel:
    """Owns an `AionLoadedModel*` handle; keeps the owning Context alive."""

    def __init__(self, ctx, ptr):
        # Strong ref: model must keep its Context alive.
        self._ctx_owner = ctx
        self._ctx_owner._register_child(self)
        self._m = ptr
        self._closed = False
        self._name_cache = {}

    @property
    def ptr(self):
        return self._m

    def _c_name(self, name: str):
        cached = self._name_cache.get(name)
        if cached is not None:
            return cached
        b = name.encode("utf-8")
        c_name = ffi.new("char[]", b)
        self._name_cache[name] = c_name
        return c_name

    @property
    def context(self):
        """Owning context for this loaded model."""

        return self._ctx_owner

    def close(self) -> None:
        if self._closed:
            return
        lib.aion_loaded_model_destroy(self._m)
        try:
            self._ctx_owner._unregister_child(self)
        except Exception:
            pass
        self._m = ffi.NULL
        self._closed = True

    def __enter__(self) -> "LoadedModel":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):  # pragma: no cover
        try:
            if not getattr(self, "_closed", True):
                lib.aion_loaded_model_destroy(getattr(self, "_m", ffi.NULL))
        except Exception:
            pass

    @classmethod
    def load(
        cls,
        ctx,
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

        out_m = ffi.new("AionLoadedModel**")
        # The Zig side expects NUL-terminated C strings.
        b = path.encode("utf-8")
        c_path = ffi.new("char[]", b)
        absolute = os.path.isabs(path)

        # cffi structs are dynamically fielded; Any silences per-field access checks.
        opts: Any = ffi.new("AionLoadModelOptions*")  # zero-initialized
        opts.auto_init_inputs = 1
        opts.auto_positions = 1 if auto_positions else 0
        if device is not None and not _is_cpu(device):
            kind, index = _normalize_device(device)
            opts.device_kind = int(kind)
            opts.device_index = int(index)
        if cache_capacity is not None and int(cache_capacity) > 0:
            opts.cache_capacity_tokens = int(cache_capacity)
            if growable:
                opts.cache_growable = 1
                opts.cache_initial_capacity_tokens = int(initial_cache_capacity)

        if absolute:
            st = lib.aion_loaded_model_load_path_absolute(ctx.ptr, c_path, opts, out_m)
            raise_for_status(st, ctx.ptr, what="aion_loaded_model_load_path_absolute")
        else:
            st = lib.aion_loaded_model_load_path(ctx.ptr, c_path, opts, out_m)
            raise_for_status(st, ctx.ptr, what="aion_loaded_model_load_path")

        return cls(ctx, out_m[0])

    def input_count(self) -> int:
        return int(lib.aion_loaded_model_input_count(self._m))

    def output_count(self) -> int:
        return int(lib.aion_loaded_model_output_count(self._m))

    def input_names(self) -> List[str]:
        n = self.input_count()
        return [_get_indexed_name(self._ctx_owner.ptr, lib.aion_loaded_model_input_name, self._m, i) for i in range(n)]

    def output_names(self) -> List[str]:
        n = self.output_count()
        return [_get_indexed_name(self._ctx_owner.ptr, lib.aion_loaded_model_output_name, self._m, i) for i in range(n)]

    def input_specs(self) -> List[TensorSpec]:
        names = self.input_names()
        out: List[TensorSpec] = []
        for i, name in enumerate(names):
            out.append(TensorSpec(name=name, dtype=self.input_dtype(i), rank=self.input_rank(i)))
        return out

    def output_specs(self) -> List[TensorSpec]:
        names = self.output_names()
        out: List[TensorSpec] = []
        for i, name in enumerate(names):
            out.append(TensorSpec(name=name, dtype=self.output_dtype(i), rank=self.output_rank(i)))
        return out

    def input_dtype(self, index: int) -> AionDType:
        out = ffi.new("AionDType*")
        st = lib.aion_loaded_model_input_dtype(self._m, int(index), out)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_loaded_model_input_dtype")
        return AionDType(int(out[0]))

    def input_rank(self, index: int) -> int:
        out = ffi.new("size_t*")
        st = lib.aion_loaded_model_input_rank(self._m, int(index), out)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_loaded_model_input_rank")
        return int(out[0])

    def output_dtype(self, index: int) -> AionDType:
        out = ffi.new("AionDType*")
        st = lib.aion_loaded_model_output_dtype(self._m, int(index), out)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_loaded_model_output_dtype")
        return AionDType(int(out[0]))

    def output_rank(self, index: int) -> int:
        out = ffi.new("size_t*")
        st = lib.aion_loaded_model_output_rank(self._m, int(index), out)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_loaded_model_output_rank")
        return int(out[0])

    def bind_input(self, name: str, tensor) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a str")
        c_name = self._c_name(name)
        st = lib.aion_loaded_model_bind_input(self._m, c_name, tensor.ptr)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_loaded_model_bind_input")

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
        c_name = self._c_name(name)
        st = lib.aion_loaded_model_set_state_input_policy(
            self._m, c_name, 1,
            int(initial_capacity), int(growth_numerator), int(growth_denominator), int(max_capacity), 0,
        )
        raise_for_status(st, self._ctx_owner.ptr, what="aion_loaded_model_set_state_input_policy")

    def set_state_input_ring(self, name: str, *, window: int) -> None:
        """Make an io-aliased recurrent-state input a fixed ring buffer of `window` tokens."""
        if not isinstance(name, str):
            raise TypeError("name must be a str")
        c_name = self._c_name(name)
        st = lib.aion_loaded_model_set_state_input_policy(self._m, c_name, 2, 0, 0, 0, 0, int(window))
        raise_for_status(st, self._ctx_owner.ptr, what="aion_loaded_model_set_state_input_policy")

    def output_is_state(self, index: int) -> bool:
        """Whether output `index` is io-aliased recurrent state (a `next_*` carry
        the runtime already writes back into its input slot every run)."""
        out = ffi.new("uint8_t*")
        st = lib.aion_loaded_model_output_is_state(self._m, int(index), out)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_loaded_model_output_is_state")
        return bool(out[0])

    @property
    def position(self) -> int:
        """Tokens consumed so far by position auto-management (0 when disabled)."""
        out = ffi.new("uint64_t*")
        st = lib.aion_loaded_model_position(self._m, out)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_loaded_model_position")
        return int(out[0])

    def set_position(self, tokens: int) -> None:
        """Overwrite the auto-tracked position (session restore / rollback)."""
        st = lib.aion_loaded_model_set_position(self._m, int(tokens))
        raise_for_status(st, self._ctx_owner.ptr, what="aion_loaded_model_set_position")

    def _default_output_names(self) -> List[str]:
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
        inputs: Mapping[str, object],
        *,
        outputs: Iterable[str] | None = None,
    ) -> dict[str, object]:
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
        inputs: Mapping[str, object],
        *,
        outputs: Iterable[str] | None = None,
    ) -> dict[str, NDArrayF32]:
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
                    t = Tensor.from_numpy(self._ctx_owner, v)
                    temps.append(t)
                self.bind_input(name, t)

            self.run()

            out_names = list(outputs) if outputs is not None else self._default_output_names()
            outs: dict[str, NDArrayF32] = {}
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

    def run(self) -> None:
        st = lib.aion_loaded_model_run(self._m)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_loaded_model_run")

    def reset_state(self) -> None:
        """Zero all recurrent (io-aliased) input state — KV caches, LSTM h/c, etc.

        Call this between independent sequences (e.g. utterances) instead of
        reloading the model. No-op before the first ``run()``.
        """
        st = lib.aion_loaded_model_reset_state(self._m)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_loaded_model_reset_state")

    def output_tensor(self, name: str):
        if not isinstance(name, str):
            raise TypeError("name must be a str")
        c_name = self._c_name(name)
        out_t = ffi.new("AionTensor**")
        st = lib.aion_loaded_model_output_tensor(self._m, c_name, out_t)
        raise_for_status(st, self._ctx_owner.ptr, what="aion_loaded_model_output_tensor")
        from .tensor import Tensor

        return Tensor._from_handle(self._ctx_owner, out_t[0])
