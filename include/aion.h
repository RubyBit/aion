/*******************************************************************************
 * Copyright (c) 2026 Angelos-Ermis Mangos
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#pragma once

// Aion C ABI (runtime-first).
//
// Goals:
// - Make it easy to call Aion from Python (ctypes/cffi) and Rust (bindgen).
// - Keep the surface small and stable.
// - Provide deterministic error codes + an optional per-context last-error message.
//
// Notes on lifetimes:
// - All tensors are owned by the Aion context's internal storage manager.
// - `aion_tensor_destroy()` frees only the *handle*, not the underlying storage.
// - Destroy all models/tensors before destroying the context.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
  #if defined(AION_SHARED)
    #if defined(AION_BUILDING)
      #define AION_API __declspec(dllexport)
    #else
      #define AION_API __declspec(dllimport)
    #endif
  #else
    #define AION_API
  #endif
#else
  #define AION_API
#endif

typedef struct AionContext AionContext;
typedef struct AionTensor AionTensor;
typedef struct AionLoadedModel AionLoadedModel;

typedef enum AionStatus {
    AION_OK = 0,
    AION_INVALID_ARGUMENT = 1,
    AION_OUT_OF_MEMORY = 2,
    AION_UNSUPPORTED = 3,
    AION_INTERNAL_ERROR = 4,
} AionStatus;

typedef enum AionDType {
    AION_DTYPE_F32 = 0,
    AION_DTYPE_F16 = 1,
    AION_DTYPE_I8 = 2,
    AION_DTYPE_Q4_0 = 3,
    AION_DTYPE_Q8_0 = 4,
  AION_DTYPE_I32 = 5,
} AionDType;

// -----------------------------------------------------------------------------
// Version / diagnostics
// -----------------------------------------------------------------------------

AION_API uint32_t aion_version_major(void);
AION_API uint32_t aion_version_minor(void);
AION_API uint32_t aion_version_patch(void);

AION_API const char* aion_status_string(AionStatus status);

// Copy the last error message associated with this context into `buf`.
// - Always writes `out_len` if non-null.
// - Writes a NUL terminator when `cap > 0`.
// - If truncated, the message is still NUL-terminated.
AION_API AionStatus aion_context_last_error_message(
    const AionContext* ctx,
    char* buf,
    size_t cap,
    size_t* out_len);

// -----------------------------------------------------------------------------
// Context
// -----------------------------------------------------------------------------

AION_API AionStatus aion_context_create_cpu(size_t thread_count, AionContext** out_ctx);
AION_API void aion_context_destroy(AionContext* ctx);

// -----------------------------------------------------------------------------
// Tensors (scalar I/O)
// -----------------------------------------------------------------------------

AION_API AionStatus aion_tensor_create_empty(
    AionContext* ctx,
    AionDType dtype,
    size_t rank,
    const size_t* shape,
    AionTensor** out_tensor);

// Like `aion_tensor_create_empty`, but with an explicit per-axis tile shape.
//
// `tile_shape` must have `rank` entries; each entry must be in `1..=shape[d]` and,
// where dtype is quantized, must align to the dtype's block granularity on the
// block axis. Use this when a specific tile layout is required by a graph op
// (e.g. KV caches consumed by `KVCacheAppend`, which require the full head_dim
// contiguous in a single tile).
AION_API AionStatus aion_tensor_create_empty_tiled(
    AionContext* ctx,
    AionDType dtype,
    size_t rank,
    const size_t* shape,
    const size_t* tile_shape,
    AionTensor** out_tensor);

// Create a tensor, optionally initializing its contents.
//
// If `values == NULL`, this is equivalent to `aion_tensor_create_empty()`.
// If `values != NULL`, `values_len` is the element count.
//
// v1 notes:
// - Initialization from `values` is supported for scalar (non-quantized) dtypes.
// - For quantized dtypes, pass `values == NULL` and `values_len == 0`.
AION_API AionStatus aion_tensor_create(
    AionContext* ctx,
    AionDType dtype,
    size_t rank,
    const size_t* shape,
    const void* values,
    size_t values_len,
    AionTensor** out_tensor);

AION_API void aion_tensor_destroy(AionTensor* t);

AION_API AionDType aion_tensor_dtype(const AionTensor* t);
AION_API size_t aion_tensor_rank(const AionTensor* t);
AION_API AionStatus aion_tensor_shape(const AionTensor* t, size_t* out_dims, size_t out_rank);

// Read tensor contents into a caller-provided buffer.
//
// - `dtype` must match `aion_tensor_dtype(t)`.
// - `out_len` is element count (not bytes).
//
// v1 notes:
// - Only scalar (non-quantized) dtypes are supported.
AION_API AionStatus aion_tensor_read(const AionTensor* t, AionDType dtype, void* out_values, size_t out_len);

// Write tensor contents from a caller-provided buffer.
//
// - `dtype` must match `aion_tensor_dtype(t)`.
// - `values_len` is element count (not bytes).
//
// v1 notes:
// - Only scalar (non-quantized) dtypes are supported.
AION_API AionStatus aion_tensor_write(AionTensor* t, AionDType dtype, const void* values, size_t values_len);

// Read a scalar tensor into a single value.
//
// - `dtype` must match `aion_tensor_dtype(t)`.
// - Tensor must contain exactly 1 element.
//
// v1 notes:
// - Only scalar (non-quantized) dtypes are supported.
AION_API AionStatus aion_tensor_read_scalar(const AionTensor* t, AionDType dtype, void* out_value);

// -----------------------------------------------------------------------------
// Loaded model runtime (.aion packages)
// -----------------------------------------------------------------------------

AION_API AionStatus aion_loaded_model_load_path(AionContext* ctx, const char* path, AionLoadedModel** out_model);
AION_API AionStatus aion_loaded_model_load_path_absolute(AionContext* ctx, const char* absolute_path, AionLoadedModel** out_model);
AION_API void aion_loaded_model_destroy(AionLoadedModel* m);

AION_API size_t aion_loaded_model_input_count(const AionLoadedModel* m);
AION_API size_t aion_loaded_model_output_count(const AionLoadedModel* m);

AION_API AionStatus aion_loaded_model_input_name(const AionLoadedModel* m, size_t index, char* buf, size_t cap, size_t* out_len);
AION_API AionStatus aion_loaded_model_output_name(const AionLoadedModel* m, size_t index, char* buf, size_t cap, size_t* out_len);

AION_API AionStatus aion_loaded_model_input_dtype(const AionLoadedModel* m, size_t index, AionDType* out_dtype);
AION_API AionStatus aion_loaded_model_input_rank(const AionLoadedModel* m, size_t index, size_t* out_rank);
AION_API AionStatus aion_loaded_model_output_dtype(const AionLoadedModel* m, size_t index, AionDType* out_dtype);
AION_API AionStatus aion_loaded_model_output_rank(const AionLoadedModel* m, size_t index, size_t* out_rank);

AION_API AionStatus aion_loaded_model_bind_input(AionLoadedModel* m, const char* name, const AionTensor* tensor);
AION_API AionStatus aion_loaded_model_run(AionLoadedModel* m);
/* Zero all recurrent (io-aliased) input state, e.g. KV caches and LSTM h/c.
   Use between independent sequences. No-op before the first run. */
AION_API AionStatus aion_loaded_model_reset_state(AionLoadedModel* m);
AION_API AionStatus aion_loaded_model_output_tensor(AionLoadedModel* m, const char* name, AionTensor** out_tensor);

#ifdef __cplusplus
}
#endif
