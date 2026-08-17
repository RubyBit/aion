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

// Device selector. A tensor/model lives on exactly one device at a time.
// For CPU, `index` is ignored; for GPU it names the registered GPU.
typedef enum AionDeviceKind {
    AION_DEVICE_CPU = 0,
    AION_DEVICE_GPU = 1,
} AionDeviceKind;

typedef enum AionGpuPower {
    AION_GPU_POWER_DEFAULT = 0,
    AION_GPU_POWER_LOW = 1,
    AION_GPU_POWER_HIGH = 2,
} AionGpuPower;

typedef enum AionGpuBackend {
    AION_GPU_BACKEND_ANY = 0,
    AION_GPU_BACKEND_VULKAN = 1,
    AION_GPU_BACKEND_D3D12 = 2,
    AION_GPU_BACKEND_METAL = 3,
    AION_GPU_BACKEND_GL = 4,
} AionGpuBackend;

// Per-GPU creation options. `adapter_index < 0` means auto (no explicit adapter).
typedef struct AionGpuOptions {
    AionGpuPower power;
    AionGpuBackend backend;
    int32_t adapter_index;
} AionGpuOptions;

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

// Create a context, optionally registering GPUs.
//
// - `gpus` points to `gpu_count` option structs (may be NULL when `gpu_count == 0`).
//   `gpus[i]` becomes the device selected by `(AION_DEVICE_GPU, i)`.
// - On a CPU-only build (compiled without `-Dgpu`), a non-zero `gpu_count`
//   returns AION_UNSUPPORTED.
// - At most 16 GPUs may be registered per call.
AION_API AionStatus aion_context_create(
    size_t thread_count,
    const AionGpuOptions* gpus,
    size_t gpu_count,
    AionContext** out_ctx);

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
// (e.g. KV caches consumed by `SequenceAppend`, which require the full head_dim
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

// Migrate a tensor to `(kind, index)` (move semantics: the source-device copy
// is freed). After moving off the CPU, host read/write fail until the tensor is
// migrated back with `aion_tensor_to(t, AION_DEVICE_CPU, 0)`. Idempotent when
// already on the target device.
AION_API AionStatus aion_tensor_to(AionTensor* t, AionDeviceKind kind, uint32_t index);

// Report the device a tensor is currently resident on. Either out-pointer may
// be NULL if that field is not needed.
AION_API AionStatus aion_tensor_device(const AionTensor* t, AionDeviceKind* out_kind, uint32_t* out_index);

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

// Load options. Pass NULL for the defaults: CPU, auto-init inputs, auto
// positions, no cache auto-sizing. When passing a struct, start from
// zero-initialized memory and set the fields you need (note the two
// enabled-by-default flags below).
// - device_kind/device_index: place the model on a device. The model's backend
//   and tile policy follow the device; bound CPU inputs are auto-migrated on
//   run and outputs are flushed back to host for reading.
// - auto_init_inputs: 1 = auto-allocate + zero unbound inputs (default when
//   opts == NULL).
// - auto_positions: 1 = auto-feed role-declared cache_write_index /
//   cache_visible_end / positions inputs from the tracked sequence position and
//   advance it per run (default when opts == NULL).
// - cache_capacity_tokens > 0: size every role-declared sequence cache whose
//   capacity axis is a free dim symbol to this many tokens.
// - cache_growable != 0: start those caches at cache_initial_capacity_tokens
//   (0 = default 8) and grow on demand (factor cache_growth_numerator /
//   cache_growth_denominator, 0/0 = default 3/2) up to cache_capacity_tokens.
typedef struct AionLoadModelOptions {
    uint32_t device_kind;
    uint32_t device_index;
    uint8_t auto_init_inputs;
    uint8_t auto_positions;
    uint8_t _pad[6];
    uint64_t cache_capacity_tokens;
    uint8_t cache_growable;
    uint8_t _pad1[7];
    uint64_t cache_initial_capacity_tokens;
    uint64_t cache_growth_numerator;
    uint64_t cache_growth_denominator;
    uint64_t plan_cache_budget_bytes;
} AionLoadModelOptions;

AION_API AionStatus aion_loaded_model_load_path(AionContext* ctx, const char* path, const AionLoadModelOptions* opts, AionLoadedModel** out_model);
AION_API AionStatus aion_loaded_model_load_path_absolute(AionContext* ctx, const char* absolute_path, const AionLoadModelOptions* opts, AionLoadedModel** out_model);

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
/* Declare a sequence-cache policy for an io-aliased recurrent-state input by
   name, so the runtime can grow (or ring-wrap) its state slot on demand instead
   of the caller pre-allocating the maximum. kind: 0=none (fixed), 1=growable,
   2=ring. Growable uses initial/growth_numerator/growth_denominator/max_capacity;
   ring uses ring_window_tokens. Unused fields are ignored per kind. */
AION_API AionStatus aion_loaded_model_set_state_input_policy(AionLoadedModel* m, const char* name, uint32_t kind, uint64_t initial_capacity_tokens, uint64_t growth_numerator, uint64_t growth_denominator, uint64_t max_capacity_tokens, uint64_t ring_window_tokens);
AION_API AionStatus aion_loaded_model_output_tensor(AionLoadedModel* m, const char* name, AionTensor** out_tensor);

/* Tokens consumed so far by position auto-management (0 when disabled). */
AION_API AionStatus aion_loaded_model_position(const AionLoadedModel* m, uint64_t* out_tokens);
/* Overwrite the auto-tracked position (session restore / rollback). */
AION_API AionStatus aion_loaded_model_set_position(AionLoadedModel* m, uint64_t tokens);
/* Whether output `index` is io-aliased recurrent state (a carry the runtime
   writes back into its input slot each run). Lets callers skip copying such
   outputs out by default. */
AION_API AionStatus aion_loaded_model_output_is_state(const AionLoadedModel* m, size_t index, uint8_t* out_is_state);

// -----------------------------------------------------------------------------
// Model authoring (Builder -> compile / export)
//
// Build a computation graph in code, then either compile it to a runnable model
// in-process (`aion_builder_compile`, reusing the AionLoadedModel run surface)
// or serialize it to a `.aion` file (`aion_builder_export_path`). Values are
// plain AionValueId (a u32 scoped to the builder), not heap handles. Every op
// goes through the single `aion_builder_op` entry, whose AionOpSpec attribute
// union mirrors the core graph op set.
// -----------------------------------------------------------------------------

typedef struct AionBuilder AionBuilder;

// Opaque id of a value inside a builder (mirrors the Zig TensorRef, a u32).
typedef uint32_t AionValueId;

typedef enum AionUnaryOp {
    AION_UNARY_RELU = 0,
    AION_UNARY_GELU = 1,
    AION_UNARY_SILU = 2,
    AION_UNARY_SIGMOID = 3,
    AION_UNARY_TANH = 4,
    AION_UNARY_SQRT = 5,
    AION_UNARY_LOG = 6,
} AionUnaryOp;

typedef enum AionBinaryOp {
    AION_BINARY_ADD = 0,
    AION_BINARY_SUB = 1,
    AION_BINARY_MUL = 2,
    AION_BINARY_DIV = 3,
    AION_BINARY_EQ = 4,
    AION_BINARY_NE = 5,
    AION_BINARY_LT = 6,
    AION_BINARY_GT = 7,
    AION_BINARY_LE = 8,
    AION_BINARY_GE = 9,
} AionBinaryOp;

typedef enum AionReduceOp {
    AION_REDUCE_SUM = 0,
    AION_REDUCE_MEAN = 1,
} AionReduceOp;

typedef enum AionPadMode {
    AION_PAD_ZERO = 0,
    AION_PAD_REFLECT = 1,
} AionPadMode;

typedef enum AionInputRoleKind {
    AION_ROLE_SEQUENCE_CACHE = 1,
    AION_ROLE_CACHE_WRITE_INDEX = 2,
    AION_ROLE_CACHE_VISIBLE_END = 3,
    AION_ROLE_POSITIONS = 4,
    AION_ROLE_TOKENS = 5,
    AION_ROLE_STATE = 6,
} AionInputRoleKind;

typedef enum AionOp {
    AION_OP_MATMUL = 0,
    AION_OP_MATMUL_NT = 1,
    AION_OP_ELEMWISE = 2,
    AION_OP_UNARY = 3,
    AION_OP_SOFTMAX = 4,
    AION_OP_LAYERNORM = 5,
    AION_OP_RMSNORM = 6,
    AION_OP_ATTENTION = 7,
    AION_OP_RELPOS_MHA = 8,
    AION_OP_CONV1D = 9,
    AION_OP_CONV2D = 10,
    AION_OP_COPY = 11,
    AION_OP_ROPE1D = 12,
    AION_OP_SEQUENCE_APPEND = 13,
    AION_OP_REDUCE = 14,
    AION_OP_CONCAT = 15,
    AION_OP_RESHAPE = 16,
    AION_OP_SQUEEZE = 17,
    AION_OP_UNSQUEEZE = 18,
    AION_OP_TRANSPOSE2D = 19,
    AION_OP_SLICE = 20,
    AION_OP_LSTM_CELL = 21,
    AION_OP_RFFT = 22,
    AION_OP_STFT = 23,
    AION_OP_CAST = 24,
    AION_OP_ARGMAX = 25,
    AION_OP_SCATTER_ROW = 26,
    AION_OP_GELU_MUL = 27,
    AION_OP_GATHER = 28,
    AION_OP_DIM = 29,
    AION_OP_IOTA = 30,
} AionOp;

// Per-op attributes. Only the member matching AionOpSpec.op is read.
typedef union AionOpAttr {
    struct { float alpha; float beta; } matmul;
    struct { AionBinaryOp op; } elemwise;
    struct { AionUnaryOp op; } unary;
    struct { int32_t axis; } softmax;
    struct { float eps; const size_t* normalized_shape; size_t normalized_shape_len; } norm;
    struct {
        float scale;
        uint8_t causal;
        size_t sliding_window;
        float attn_logits_soft_cap;
        uint8_t has_query_positions;
        uint8_t has_kv_lengths;
    } attention;
    // The head count is q's dim 2 (q is [B, T, heads, D]) — never passed separately.
    //
    // chunk_size = 0 means attend to every key; otherwise a query attends to its
    // own chunk of `chunk_size` keys plus `chunk_left` keys before that chunk's
    // start (NeMo "chunked_limited"). Structural, so it replaces an additive
    // [T_q, T_kv] mask and lets the kernels skip out-of-window keys.
    struct { float scale; size_t chunk_size; size_t chunk_left; } relpos_mha;
    struct { size_t stride; size_t dilation; size_t pad_left; size_t pad_right; size_t groups; AionPadMode pad_mode; } conv1d;
    struct { size_t stride_h; size_t stride_w; size_t dilation_h; size_t dilation_w; size_t pad_top; size_t pad_bottom; size_t pad_left; size_t pad_right; size_t groups; AionPadMode pad_mode; } conv2d;
    struct { float base_frequency; float scale_factor; float rope_proportion; } rope1d;
    struct { AionReduceOp op; int32_t axis; uint8_t has_axis; } reduce;
    struct { int32_t axis; } concat;
    // reshape/slice dims may be symbolic: `*_symbols`, when non-NULL, is an array
    // (length shape_len/len) whose entry is a NUL-terminated dim-symbol name for a
    // symbolic axis or NULL for a constant axis (whose size is shape[i]/lens[i],
    // the authoring placeholder). NULL for the whole array means all-constant.
    struct { const size_t* shape; size_t shape_len; const char* const* shape_symbols; } reshape;
    struct { int32_t axis; uint8_t has_axis; } squeeze;
    struct { int32_t axis; } unsqueeze;
    struct { const size_t* starts; const size_t* lens; size_t len; const char* const* len_symbols; } slice;
    struct { size_t n_fft; size_t hop_length; uint8_t center; } stft;
    struct { AionDType to_dtype; } cast;
    struct { int32_t axis; } argmax;
    struct { int32_t axis; size_t batch_dims; } gather;
} AionOpAttr;

// A single op to append: which op, its input value ids (like a graph node's
// inputs), and its attributes.
typedef struct AionOpSpec {
    AionOp op;
    const AionValueId* inputs;
    size_t inputs_len;
    AionOpAttr attr;
} AionOpSpec;

AION_API AionStatus aion_builder_create(AionContext* ctx, AionBuilder** out_builder);
AION_API void aion_builder_destroy(AionBuilder* b);

AION_API AionStatus aion_builder_input(AionBuilder* b, AionDType dtype, size_t rank, const size_t* shape, AionValueId* out_value);
// Bind an already-created owned tensor (float or quantized) as a graph weight.
AION_API AionStatus aion_builder_param(AionBuilder* b, const AionTensor* tensor, AionValueId* out_value);
AION_API AionStatus aion_builder_name(AionBuilder* b, AionValueId value, const char* name);

// Bind a weight under a semantic, scope-qualified name (`layers.3/attn/weight`).
// Unlike aion_builder_param, whose generated name is positional and shifts when
// construction order changes, this is the stable key load/swap-by-name uses.
AION_API AionStatus aion_builder_param_named(AionBuilder* b, const AionTensor* tensor, const char* name, AionValueId* out_value);

// Scopes nest, so a module tree produces `state_dict`-style parameter paths.
// begin_* hands back the depth it opened at; pass that to end_scope, which closes
// every scope at or below it (so an early return cannot leak a level).
AION_API AionStatus aion_builder_begin_scope(AionBuilder* b, const char* name, size_t* out_depth);
// Opens `{base}#{n}`, numbered per parent scope; writes the resolved segment.
AION_API AionStatus aion_builder_begin_auto_scope(AionBuilder* b, const char* base, size_t* out_depth, char* buf, size_t cap, size_t* out_len);
AION_API AionStatus aion_builder_end_scope(AionBuilder* b, size_t depth);
AION_API AionStatus aion_builder_scope_path(const AionBuilder* b, char* buf, size_t cap, size_t* out_len);

// Several ops require an operand where the maths wants a number. These bind (and
// cache) that operand, so needing the same identity vector or scalar in many
// layers costs one parameter.
AION_API AionStatus aion_builder_constant(AionBuilder* b, float value, AionValueId* out_value);
// `fill` must be 0.0 (a norm's identity beta) or 1.0 (its identity gamma).
AION_API AionStatus aion_builder_filled_vec(AionBuilder* b, size_t dim, float fill, AionValueId* out_value);

// 0 = not a bound parameter, 1 = user-supplied weight, 2 = synthesized constant.
AION_API AionStatus aion_builder_param_kind(const AionBuilder* b, AionValueId value, uint32_t* out_kind);
AION_API AionStatus aion_builder_value_name(const AionBuilder* b, AionValueId value, char* buf, size_t cap, size_t* out_len);
AION_API AionStatus aion_builder_has_param_named(const AionBuilder* b, const char* name, uint8_t* out_found);

// Authoring-time shape/dtype introspection (eager per-op inference keeps every
// value's shape current). Shapes reflect authoring placeholder sizes: an axis
// declared dynamic reports the size it was declared with, propagated to derived
// values. Mirrors the tensor rank/shape/dtype readers.
AION_API AionStatus aion_builder_value_rank(const AionBuilder* b, AionValueId value, size_t* out_rank);
AION_API AionStatus aion_builder_value_shape(const AionBuilder* b, AionValueId value, size_t* out_dims, size_t out_rank);
AION_API AionStatus aion_builder_value_dtype(const AionBuilder* b, AionValueId value, AionDType* out_dtype);
// The dim symbol on `axis`, or an empty name when that axis has a fixed size. An
// axis is free because some input declared it so (aion_builder_add_dim_symbol) and
// inference carried the symbol to everything derived from it, so a caller can
// rebuild a shape without freezing an axis it was never told about.
AION_API AionStatus aion_builder_value_dim_symbol(const AionBuilder* b, AionValueId value, size_t axis, char* buf, size_t cap, size_t* out_len);
// The authoring placeholder size a dim symbol was declared with. A free axis still
// needs a concrete size to author against; asking here keeps that size recorded in
// one place instead of mirrored by every caller.
AION_API AionStatus aion_builder_symbol_size(const AionBuilder* b, const char* name, size_t* out_size);

// Append one op. Reads only the AionOpAttr member matching `spec->op`; variable
// arity (concat, optional bias/mask) is carried by `spec->inputs`.
AION_API AionStatus aion_builder_op(AionBuilder* b, const AionOpSpec* spec, AionValueId* out_value);

// Control flow. Build a region (a branch body or loop body) between
// begin/end_region, then reference it from `if`/`loop`. `end_region` returns a
// region id. `loop` writes `carried_len` final carried values into `out_values`
// (single-carry loops pass carried_len == 1); `has_cond_carry == 0` means no
// continue-predicate carry.
typedef uint32_t AionRegionId;
AION_API AionStatus aion_builder_begin_region(AionBuilder* b);
AION_API AionStatus aion_builder_end_region(AionBuilder* b, const AionValueId* outputs, size_t outputs_len, AionRegionId* out_region);
AION_API AionStatus aion_builder_if(AionBuilder* b, AionValueId cond, AionRegionId then_region, AionRegionId else_region, AionValueId* out_value);
AION_API AionStatus aion_builder_loop(AionBuilder* b, const AionValueId* carried, size_t carried_len, AionRegionId body_region, size_t trip, size_t cond_carry, uint8_t has_cond_carry, uint8_t check_before, AionValueId* out_values);

// Declarations consumed by compile/export. Marking an output also names it.
AION_API AionStatus aion_builder_mark_output(AionBuilder* b, AionValueId value, const char* name);
// Forget every output marked so far. `mark_output` accumulates, so a caller that
// means "compile exactly these outputs" clears first.
AION_API AionStatus aion_builder_clear_outputs(AionBuilder* b);
AION_API AionStatus aion_builder_add_output_alias(AionBuilder* b, AionValueId input_value, AionValueId output_value);
AION_API AionStatus aion_builder_add_input_role(AionBuilder* b, AionValueId value, AionInputRoleKind kind, int32_t axis, uint8_t has_axis, const char* capacity_symbol, uint8_t zero_init, uint8_t allow_growable, uint8_t allow_ring);
AION_API AionStatus aion_builder_add_dim_symbol(AionBuilder* b, AionValueId value, size_t axis, const char* name);
AION_API AionStatus aion_builder_add_metadata(AionBuilder* b, const char* key, const char* value);

// Terminals. `compile` builds an in-process runnable model (concrete shapes
// only). `export_path` serializes a `.aion` (supports dim symbols). Both consume
// the marked outputs + accumulated aliases/roles/metadata.
AION_API AionStatus aion_builder_compile(AionBuilder* b, AionDeviceKind device_kind, uint32_t device_index, AionLoadedModel** out_model);
AION_API AionStatus aion_builder_export_path(AionBuilder* b, const char* path);
AION_API AionStatus aion_builder_export_path_absolute(AionBuilder* b, const char* path);

// Quantized-tensor creation. `aion_tensor_quantize` packs row-major f32 values
// into a quantized tensor (q8_0 today); `aion_tensor_create_quant` ingests
// pre-packed quant bytes.
// `quant_axis` selects the block axis (every block_elems along it forms one
// block). For a matmul-B weight this is the K reduction axis (rank-2); for an
// embedding table blocked along the feature dim it is the last axis.
AION_API AionStatus aion_tensor_quantize(AionContext* ctx, AionDType dtype, size_t rank, const size_t* shape, size_t quant_axis, const float* values, size_t values_len, AionTensor** out_tensor);
AION_API AionStatus aion_tensor_create_quant(AionContext* ctx, AionDType dtype, size_t rank, const size_t* shape, size_t quant_axis, const uint8_t* packed, size_t packed_len, AionTensor** out_tensor);

#ifdef __cplusplus
}
#endif
