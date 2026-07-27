/* SPDX-License-Identifier: Apache-2.0 */
/* Minimal Aion C ABI declarations for CFFI (API mode).

   This is intentionally macro-free and self-contained so CFFI/pycparser can
   parse it reliably.

   NOTE: This header is for *FFI parsing only*. The actual C compilation still
   includes the real `include/aion.h` via `set_source()`.
*/

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

typedef struct AionGpuOptions {
    AionGpuPower power;
    AionGpuBackend backend;
    int32_t adapter_index;
} AionGpuOptions;

uint32_t aion_version_major(void);
uint32_t aion_version_minor(void);
uint32_t aion_version_patch(void);

const char* aion_status_string(AionStatus status);

AionStatus aion_context_last_error_message(
    const AionContext* ctx,
    char* buf,
    size_t cap,
    size_t* out_len);

AionStatus aion_context_create_cpu(size_t thread_count, AionContext** out_ctx);
AionStatus aion_context_create(
    size_t thread_count,
    const AionGpuOptions* gpus,
    size_t gpu_count,
    AionContext** out_ctx);
void aion_context_destroy(AionContext* ctx);

AionStatus aion_tensor_create_empty(
    AionContext* ctx,
    AionDType dtype,
    size_t rank,
    const size_t* shape,
    AionTensor** out_tensor);

AionStatus aion_tensor_create_empty_tiled(
    AionContext* ctx,
    AionDType dtype,
    size_t rank,
    const size_t* shape,
    const size_t* tile_shape,
    AionTensor** out_tensor);

AionStatus aion_tensor_create(
    AionContext* ctx,
    AionDType dtype,
    size_t rank,
    const size_t* shape,
    const void* values,
    size_t values_len,
    AionTensor** out_tensor);

void aion_tensor_destroy(AionTensor* t);

AionStatus aion_tensor_to(AionTensor* t, AionDeviceKind kind, uint32_t index);
AionStatus aion_tensor_device(const AionTensor* t, AionDeviceKind* out_kind, uint32_t* out_index);

AionDType aion_tensor_dtype(const AionTensor* t);
size_t aion_tensor_rank(const AionTensor* t);
AionStatus aion_tensor_shape(const AionTensor* t, size_t* out_dims, size_t out_rank);

AionStatus aion_tensor_read(const AionTensor* t, AionDType dtype, void* out_values, size_t out_len);
AionStatus aion_tensor_write(AionTensor* t, AionDType dtype, const void* values, size_t values_len);
AionStatus aion_tensor_read_scalar(const AionTensor* t, AionDType dtype, void* out_value);

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
} AionLoadModelOptions;

AionStatus aion_loaded_model_load_path(AionContext* ctx, const char* path, const AionLoadModelOptions* opts, AionLoadedModel** out_model);
AionStatus aion_loaded_model_load_path_absolute(AionContext* ctx, const char* absolute_path, const AionLoadModelOptions* opts, AionLoadedModel** out_model);
void aion_loaded_model_destroy(AionLoadedModel* m);

size_t aion_loaded_model_input_count(const AionLoadedModel* m);
size_t aion_loaded_model_output_count(const AionLoadedModel* m);

AionStatus aion_loaded_model_input_name(const AionLoadedModel* m, size_t index, char* buf, size_t cap, size_t* out_len);
AionStatus aion_loaded_model_output_name(const AionLoadedModel* m, size_t index, char* buf, size_t cap, size_t* out_len);

AionStatus aion_loaded_model_input_dtype(const AionLoadedModel* m, size_t index, AionDType* out_dtype);
AionStatus aion_loaded_model_input_rank(const AionLoadedModel* m, size_t index, size_t* out_rank);
AionStatus aion_loaded_model_output_dtype(const AionLoadedModel* m, size_t index, AionDType* out_dtype);
AionStatus aion_loaded_model_output_rank(const AionLoadedModel* m, size_t index, size_t* out_rank);

AionStatus aion_loaded_model_bind_input(AionLoadedModel* m, const char* name, const AionTensor* tensor);
AionStatus aion_loaded_model_run(AionLoadedModel* m);
AionStatus aion_loaded_model_reset_state(AionLoadedModel* m);
AionStatus aion_loaded_model_set_state_input_policy(AionLoadedModel* m, const char* name, uint32_t kind, uint64_t initial_capacity_tokens, uint64_t growth_numerator, uint64_t growth_denominator, uint64_t max_capacity_tokens, uint64_t ring_window_tokens);
AionStatus aion_loaded_model_output_tensor(AionLoadedModel* m, const char* name, AionTensor** out_tensor);
AionStatus aion_loaded_model_position(const AionLoadedModel* m, uint64_t* out_tokens);
AionStatus aion_loaded_model_set_position(AionLoadedModel* m, uint64_t tokens);
AionStatus aion_loaded_model_output_is_state(const AionLoadedModel* m, size_t index, uint8_t* out_is_state);

/* Model authoring (Builder -> compile / export). */

typedef struct AionBuilder AionBuilder;

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
    AION_OP_BROADCAST_LAST_DIM = 3,
    AION_OP_UNARY = 4,
    AION_OP_SOFTMAX = 5,
    AION_OP_LAYERNORM = 6,
    AION_OP_RMSNORM = 7,
    AION_OP_ATTENTION = 8,
    AION_OP_MHA = 9,
    AION_OP_MHA_CACHED = 10,
    AION_OP_RELPOS_MHA = 11,
    AION_OP_CONV1D = 12,
    AION_OP_CONV2D = 13,
    AION_OP_COPY = 14,
    AION_OP_GATHER_ROWS = 15,
    AION_OP_ROPE1D = 16,
    AION_OP_SEQUENCE_APPEND = 17,
    AION_OP_REDUCE = 18,
    AION_OP_CONCAT = 19,
    AION_OP_RESHAPE = 20,
    AION_OP_SQUEEZE = 21,
    AION_OP_UNSQUEEZE = 22,
    AION_OP_TRANSPOSE2D = 23,
    AION_OP_SLICE = 24,
    AION_OP_LSTM_CELL = 25,
    AION_OP_RFFT = 26,
    AION_OP_STFT = 27,
    AION_OP_CAST = 28,
    AION_OP_ARGMAX = 29,
    AION_OP_SCATTER_ROW = 30,
    AION_OP_GELU_MUL = 31,
} AionOp;

typedef union AionOpAttr {
    struct { float alpha; float beta; } matmul;
    struct { AionBinaryOp op; } elemwise;
    struct { AionUnaryOp op; } unary;
    struct { int32_t axis; } softmax;
    struct { float eps; const size_t* normalized_shape; size_t normalized_shape_len; } norm;
    struct { float scale; uint8_t causal; size_t heads; } attention;
    struct { float scale; uint8_t causal; size_t sliding_window; float attn_logits_soft_cap; } mha_cached;
    struct { float scale; size_t heads; } relpos_mha;
    struct { size_t stride; size_t dilation; size_t pad_left; size_t pad_right; size_t groups; AionPadMode pad_mode; } conv1d;
    struct { size_t stride_h; size_t stride_w; size_t dilation_h; size_t dilation_w; size_t pad_top; size_t pad_bottom; size_t pad_left; size_t pad_right; size_t groups; AionPadMode pad_mode; } conv2d;
    struct { float base_frequency; float scale_factor; float rope_proportion; } rope1d;
    struct { AionReduceOp op; int32_t axis; uint8_t has_axis; } reduce;
    struct { int32_t axis; } concat;
    struct { const size_t* shape; size_t shape_len; const char* const* shape_symbols; } reshape;
    struct { int32_t axis; uint8_t has_axis; } squeeze;
    struct { int32_t axis; } unsqueeze;
    struct { const size_t* starts; const size_t* lens; size_t len; const char* const* len_symbols; } slice;
    struct { size_t n_fft; size_t hop_length; uint8_t center; } stft;
    struct { AionDType to_dtype; } cast;
    struct { int32_t axis; } argmax;
} AionOpAttr;

typedef struct AionOpSpec {
    AionOp op;
    const AionValueId* inputs;
    size_t inputs_len;
    AionOpAttr attr;
} AionOpSpec;

AionStatus aion_builder_create(AionContext* ctx, AionBuilder** out_builder);
void aion_builder_destroy(AionBuilder* b);

AionStatus aion_builder_input(AionBuilder* b, AionDType dtype, size_t rank, const size_t* shape, AionValueId* out_value);
AionStatus aion_builder_param(AionBuilder* b, const AionTensor* tensor, AionValueId* out_value);
AionStatus aion_builder_name(AionBuilder* b, AionValueId value, const char* name);
AionStatus aion_builder_param_named(AionBuilder* b, const AionTensor* tensor, const char* name, AionValueId* out_value);
AionStatus aion_builder_begin_scope(AionBuilder* b, const char* name, size_t* out_depth);
AionStatus aion_builder_begin_auto_scope(AionBuilder* b, const char* base, size_t* out_depth, char* buf, size_t cap, size_t* out_len);
AionStatus aion_builder_end_scope(AionBuilder* b, size_t depth);
AionStatus aion_builder_scope_path(const AionBuilder* b, char* buf, size_t cap, size_t* out_len);
AionStatus aion_builder_constant(AionBuilder* b, float value, AionValueId* out_value);
AionStatus aion_builder_filled_vec(AionBuilder* b, size_t dim, float fill, AionValueId* out_value);
AionStatus aion_builder_param_kind(const AionBuilder* b, AionValueId value, uint32_t* out_kind);
AionStatus aion_builder_value_name(const AionBuilder* b, AionValueId value, char* buf, size_t cap, size_t* out_len);
AionStatus aion_builder_has_param_named(const AionBuilder* b, const char* name, uint8_t* out_found);

AionStatus aion_builder_value_rank(const AionBuilder* b, AionValueId value, size_t* out_rank);
AionStatus aion_builder_value_shape(const AionBuilder* b, AionValueId value, size_t* out_dims, size_t out_rank);
AionStatus aion_builder_value_dtype(const AionBuilder* b, AionValueId value, AionDType* out_dtype);
AionStatus aion_builder_value_dim_symbol(const AionBuilder* b, AionValueId value, size_t axis, char* buf, size_t cap, size_t* out_len);
AionStatus aion_builder_symbol_size(const AionBuilder* b, const char* name, size_t* out_size);

AionStatus aion_builder_op(AionBuilder* b, const AionOpSpec* spec, AionValueId* out_value);

typedef uint32_t AionRegionId;
AionStatus aion_builder_begin_region(AionBuilder* b);
AionStatus aion_builder_end_region(AionBuilder* b, const AionValueId* outputs, size_t outputs_len, AionRegionId* out_region);
AionStatus aion_builder_if(AionBuilder* b, AionValueId cond, AionRegionId then_region, AionRegionId else_region, AionValueId* out_value);
AionStatus aion_builder_loop(AionBuilder* b, const AionValueId* carried, size_t carried_len, AionRegionId body_region, size_t trip, size_t cond_carry, uint8_t has_cond_carry, uint8_t check_before, AionValueId* out_values);

AionStatus aion_builder_mark_output(AionBuilder* b, AionValueId value, const char* name);
AionStatus aion_builder_clear_outputs(AionBuilder* b);
AionStatus aion_builder_add_output_alias(AionBuilder* b, AionValueId input_value, AionValueId output_value);
AionStatus aion_builder_add_input_role(AionBuilder* b, AionValueId value, AionInputRoleKind kind, int32_t axis, uint8_t has_axis, const char* capacity_symbol, uint8_t zero_init, uint8_t allow_growable, uint8_t allow_ring);
AionStatus aion_builder_add_dim_symbol(AionBuilder* b, AionValueId value, size_t axis, const char* name);
AionStatus aion_builder_add_metadata(AionBuilder* b, const char* key, const char* value);

AionStatus aion_builder_compile(AionBuilder* b, AionDeviceKind device_kind, uint32_t device_index, AionLoadedModel** out_model);
AionStatus aion_builder_export_path(AionBuilder* b, const char* path);
AionStatus aion_builder_export_path_absolute(AionBuilder* b, const char* path);

AionStatus aion_tensor_quantize(AionContext* ctx, AionDType dtype, size_t rank, const size_t* shape, size_t quant_axis, const float* values, size_t values_len, AionTensor** out_tensor);
AionStatus aion_tensor_create_quant(AionContext* ctx, AionDType dtype, size_t rank, const size_t* shape, size_t quant_axis, const uint8_t* packed, size_t packed_len, AionTensor** out_tensor);
