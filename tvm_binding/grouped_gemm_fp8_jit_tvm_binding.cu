#include "tvm_binding_utils.h"

void GroupedGemmFp8Run(DLTensor* int_workspace_buffer, DLTensor* float_workspace_buffer,
                       DLTensor* A, DLTensor* B, DLTensor* SFA, DLTensor* SFB, DLTensor* D,
                       DLTensor* m_indptr, int64_t n, int64_t k, int64_t scale_granularity_m,
                       int64_t scale_granularity_n, int64_t scale_granularity_k,
                       int64_t scale_major_mode, int64_t mma_sm, TVMStreamHandle cuda_stream);

// NOTE: This registration will be overwritten by the specialized wrapper in the generated code.
TVM_FFI_DLL_EXPORT_TYPED_FUNC(grouped_gemm_fp8_run, GroupedGemmFp8Run);
