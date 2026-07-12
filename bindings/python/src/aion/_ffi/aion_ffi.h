/* SPDX-License-Identifier: Apache-2.0 */
/* Minimal Aion C ABI declarations for CFFI (API mode).

   This is intentionally macro-free and self-contained so CFFI/pycparser can
   parse it reliably.

   NOTE: This header is for *FFI parsing only*. The actual C compilation still
   includes the real `include/aion.h` via `set_source()`.
*/

typedef unsigned long long size_t;
typedef unsigned int uint32_t;
typedef int int32_t;

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
