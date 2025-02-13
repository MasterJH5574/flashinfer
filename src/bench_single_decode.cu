/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <thrust/device_vector.h>

#include <flashinfer/decode.cuh>
#include <nvbench/nvbench.cuh>

using flashinfer::QKVLayout;
using flashinfer::RotaryMode;

template <typename dtype_in, typename dtype_out, size_t rotary_mode, size_t layout>
void bench_flashinfer_single_decode(nvbench::state& state) {
  size_t seq_len = state.get_int64("seq_len");
  size_t num_qo_heads = state.get_int64("num_qo_heads");
  size_t num_kv_heads = state.get_int64("num_kv_heads");
  size_t head_dim = state.get_int64("head_dim");
  bool cooperative = state.get_int64("cooperative");
  // Allocate input data:
  thrust::device_vector<dtype_in> Q(num_qo_heads * head_dim);
  thrust::device_vector<dtype_in> K(seq_len * num_kv_heads * head_dim);
  thrust::device_vector<dtype_in> V(seq_len * num_kv_heads * head_dim);
  thrust::device_vector<dtype_out> O(num_qo_heads * head_dim);
  thrust::device_vector<float> tmp(512 * num_qo_heads * head_dim);

  // Provide throughput information:
  state.add_global_memory_reads<dtype_in>(
      num_qo_heads * head_dim + 2 * seq_len * num_kv_heads * head_dim, "Read");
  state.add_global_memory_writes<dtype_out>(num_qo_heads * head_dim, "Write");

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    timer.start();
    cudaError_t status = flashinfer::SingleDecodeWithKVCache(
        thrust::raw_pointer_cast(Q.data()), thrust::raw_pointer_cast(K.data()),
        thrust::raw_pointer_cast(V.data()), thrust::raw_pointer_cast(O.data()),
        cooperative ? thrust::raw_pointer_cast(tmp.data()) : nullptr, num_qo_heads, num_kv_heads,
        seq_len, head_dim, QKVLayout(layout), RotaryMode(rotary_mode), 1.f, 1e4,
        launch.get_stream());
    if (status != cudaSuccess) {
      state.skip("CUDA error: " + std::string(cudaGetErrorString(status)));
    }
    timer.stop();
  });
}

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define BENCH_FLASHINFER_SINGLE_DECODE(dtype_in, dtype_out, rotary_mode, layout)                \
  auto bench_flashinfer_single_decode_##dtype_in##_##dtype_out##_##rotary_mode##_##layout##_ =  \
      bench_flashinfer_single_decode<dtype_in, dtype_out, rotary_mode, layout>;                 \
  NVBENCH_BENCH(                                                                                \
      bench_flashinfer_single_decode_##dtype_in##_##dtype_out##_##rotary_mode##_##layout##_)    \
      .set_name(("bench_flashinfer_single_decode_" STR(dtype_in) "_" STR(dtype_out) "_") +      \
                flashinfer::RotaryModeToString(RotaryMode(rotary_mode)) + "_" +                 \
                flashinfer::QKVLayoutToString(QKVLayout(layout)))                               \
      .add_int64_axis("seq_len", {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}) \
      .add_int64_axis("num_qo_heads", {32})                                                     \
      .add_int64_axis("num_kv_heads", {32, 4})                                                  \
      .add_int64_axis("head_dim", {128})                                                        \
      .add_int64_axis("cooperative", {1})

BENCH_FLASHINFER_SINGLE_DECODE(half, half, 0U, 0U);
BENCH_FLASHINFER_SINGLE_DECODE(half, half, 0U, 1U);
BENCH_FLASHINFER_SINGLE_DECODE(half, half, 1U, 0U);
BENCH_FLASHINFER_SINGLE_DECODE(half, half, 1U, 1U);
BENCH_FLASHINFER_SINGLE_DECODE(__nv_fp8_e5m2, half, 0U, 0U);
BENCH_FLASHINFER_SINGLE_DECODE(__nv_fp8_e5m2, half, 0U, 1U);
BENCH_FLASHINFER_SINGLE_DECODE(__nv_fp8_e5m2, half, 1U, 0U);
BENCH_FLASHINFER_SINGLE_DECODE(__nv_fp8_e5m2, half, 1U, 1U);
