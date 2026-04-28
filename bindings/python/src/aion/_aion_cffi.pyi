# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Protocol


class CData(Protocol):
    """Opaque CFFI cdata handle (pointer/array/struct/etc)."""

    def __getitem__(self, index: int) -> Any: ...
    def __setitem__(self, index: int, value: Any) -> None: ...


class FFI(Protocol):
    """Subset of `cffi.FFI` used by these bindings."""

    NULL: CData

    def new(self, cdecl: str, init: Any = ...) -> CData: ...
    def string(self, cdata: CData) -> bytes: ...
    def from_buffer(self, cdecl: str, python_buffer: Any) -> CData: ...


ffi: FFI


class Lib(Protocol):
    # Version / diagnostics
    def aion_version_major(self) -> int: ...
    def aion_version_minor(self) -> int: ...
    def aion_version_patch(self) -> int: ...

    def aion_status_string(self, status: int) -> CData: ...

    def aion_context_last_error_message(
        self,
        ctx: CData,
        buf: CData,
        cap: int,
        out_len: CData,
    ) -> int: ...

    # Context
    def aion_context_create_cpu(self, thread_count: int, out_ctx: CData) -> int: ...
    def aion_context_destroy(self, ctx: CData) -> None: ...

    # Tensors
    def aion_tensor_create_empty(
        self,
        ctx: CData,
        dtype: int,
        rank: int,
        shape: CData,
        out_tensor: CData,
    ) -> int: ...

    def aion_tensor_create(
        self,
        ctx: CData,
        dtype: int,
        rank: int,
        shape: CData,
        values: CData,
        values_len: int,
        out_tensor: CData,
    ) -> int: ...

    def aion_tensor_create_empty_tiled(
        self,
        ctx: CData,
        dtype: int,
        rank: int,
        shape: CData,
        tile_size: CData,
        out_tensor: CData,
    ) -> int: ...

    def aion_tensor_destroy(self, t: CData) -> None: ...
    def aion_tensor_dtype(self, t: CData) -> int: ...
    def aion_tensor_rank(self, t: CData) -> int: ...
    def aion_tensor_shape(self, t: CData, out_dims: CData, out_rank: int) -> int: ...

    def aion_tensor_read(
        self,
        t: CData,
        dtype: int,
        out_values: CData,
        out_len: int,
    ) -> int: ...

    def aion_tensor_write(
        self,
        t: CData,
        dtype: int,
        values: CData,
        values_len: int,
    ) -> int: ...

    def aion_tensor_read_scalar(self, t: CData, dtype: int, out_value: CData) -> int: ...

    # Loaded model runtime
    def aion_loaded_model_load_path(self, ctx: CData, path: CData, out_model: CData) -> int: ...
    def aion_loaded_model_load_path_absolute(self, ctx: CData, absolute_path: CData, out_model: CData) -> int: ...
    def aion_loaded_model_destroy(self, m: CData) -> None: ...

    def aion_loaded_model_input_count(self, m: CData) -> int: ...
    def aion_loaded_model_output_count(self, m: CData) -> int: ...

    def aion_loaded_model_input_name(
        self,
        m: CData,
        index: int,
        buf: CData,
        cap: int,
        out_len: CData,
    ) -> int: ...

    def aion_loaded_model_output_name(
        self,
        m: CData,
        index: int,
        buf: CData,
        cap: int,
        out_len: CData,
    ) -> int: ...

    def aion_loaded_model_input_dtype(self, m: CData, index: int, out_dtype: CData) -> int: ...
    def aion_loaded_model_input_rank(self, m: CData, index: int, out_rank: CData) -> int: ...
    def aion_loaded_model_output_dtype(self, m: CData, index: int, out_dtype: CData) -> int: ...
    def aion_loaded_model_output_rank(self, m: CData, index: int, out_rank: CData) -> int: ...

    def aion_loaded_model_bind_input(self, m: CData, name: CData, tensor: CData) -> int: ...
    def aion_loaded_model_run(self, m: CData) -> int: ...
    def aion_loaded_model_output_tensor(self, m: CData, name: CData, out_tensor: CData) -> int: ...


lib: Lib
