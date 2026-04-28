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

AionDType aion_tensor_dtype(const AionTensor* t);
size_t aion_tensor_rank(const AionTensor* t);
AionStatus aion_tensor_shape(const AionTensor* t, size_t* out_dims, size_t out_rank);

AionStatus aion_tensor_read(const AionTensor* t, AionDType dtype, void* out_values, size_t out_len);
AionStatus aion_tensor_write(AionTensor* t, AionDType dtype, const void* values, size_t values_len);
AionStatus aion_tensor_read_scalar(const AionTensor* t, AionDType dtype, void* out_value);

AionStatus aion_loaded_model_load_path(AionContext* ctx, const char* path, AionLoadedModel** out_model);
AionStatus aion_loaded_model_load_path_absolute(AionContext* ctx, const char* absolute_path, AionLoadedModel** out_model);
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
AionStatus aion_loaded_model_output_tensor(AionLoadedModel* m, const char* name, AionTensor** out_tensor);
