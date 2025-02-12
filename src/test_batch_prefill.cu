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
#include <gtest/gtest.h>

#include <cstdint>
#include <flashinfer/prefill.cuh>

#include "cpu_reference.h"
#include "utils.h"

using namespace flashinfer;

template <typename T>
void _TestBatchPrefillKernelOneHotCorrectness(size_t num_kv_heads, size_t head_dim, bool causal,
                                              RotaryMode rotary_mode) {
  size_t num_qo_heads = num_kv_heads;
  uint32_t num_layers = 32;
  uint32_t page_size = 16;
  uint32_t batch_size = 4;
  std::vector<std::vector<std::vector<T>>> keys, values;
  std::vector<int32_t> q_lens(batch_size), kv_lens(batch_size);
  utils::vec_randint_(q_lens, 1, 15);
  utils::vec_randint_(kv_lens, 15, 27);
  std::vector<int32_t> append_indptr{0};
  for (size_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    append_indptr.push_back(append_indptr.back() + kv_lens[request_idx]);
  }
  std::vector<T> kv_data;
  std::vector<int32_t> kv_indptr{0};
  std::vector<int32_t> kv_indices;
  std::vector<int32_t> kv_last_page_offset;
  size_t page_counter = 0;
  for (uint32_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    std::vector<std::vector<T>> keys_layer, values_layer;
    for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
      size_t kv_len = kv_lens[request_idx];
      size_t num_pages = (kv_len + page_size - 1) / page_size;
      size_t last_page_offset = (kv_len - 1) % page_size + 1;
      std::vector<T> k(kv_len * num_kv_heads * head_dim), v(kv_len * num_kv_heads * head_dim);
      utils::vec_normal_(k);
      utils::vec_normal_(v);
      keys_layer.push_back(k);
      values_layer.push_back(v);
      kv_last_page_offset.push_back(last_page_offset);
      kv_indptr.push_back(kv_indptr.back() + num_pages);
      for (size_t j = 0; j < num_pages; ++j) {
        kv_indices.push_back(page_counter++);
      }
    }
    keys.push_back(keys_layer);
    values.push_back(values_layer);
  }

  kv_data.resize(page_counter * num_layers * 2 * num_kv_heads * page_size * head_dim);
  flashinfer::paged_kv_t<T, int32_t> paged_kv_cpu(num_layers, 0, num_kv_heads, page_size, head_dim,
                                                  batch_size, kv_data.data(), kv_indptr.data(),
                                                  kv_indices.data(), kv_last_page_offset.data());
  for (uint32_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    paged_kv_cpu.layer_idx = layer_idx;
    cpu_reference::append_paged_kv_cache<T, int32_t>(paged_kv_cpu, keys[layer_idx],
                                                     values[layer_idx], append_indptr);
  }

  // copy data to device
  thrust::device_vector<T> kv_data_device(kv_data);
  thrust::device_vector<int32_t> kv_indptr_device(kv_indptr);
  thrust::device_vector<int32_t> kv_indices_device(kv_indices);
  thrust::device_vector<int32_t> kv_last_page_offset_device(kv_last_page_offset);

  // create paged_kv object
  flashinfer::paged_kv_t<T, int32_t> paged_kv = paged_kv_cpu;
  paged_kv.data = thrust::raw_pointer_cast(kv_data_device.data());
  paged_kv.indptr = thrust::raw_pointer_cast(kv_indptr_device.data());
  paged_kv.indices = thrust::raw_pointer_cast(kv_indices_device.data());
  paged_kv.last_page_offset = thrust::raw_pointer_cast(kv_last_page_offset_device.data());

  for (uint32_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    paged_kv.layer_idx = layer_idx;
    for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
      // create one-hot queries
      int32_t q_len = q_lens[request_idx], kv_len = kv_lens[request_idx];
      std::vector<int32_t> q_indptr{0};
      for (uint32_t i = 0; i < batch_size; ++i) {
        q_indptr.push_back(i >= request_idx ? q_len : 0);
      }
      std::vector<T> q(q_len * num_qo_heads * head_dim);
      utils::vec_normal_(q);

      std::vector<T> o_ref = cpu_reference::single_mha<T, T>(
          q, keys[layer_idx][request_idx], values[layer_idx][request_idx], q_len, kv_len,
          num_qo_heads, num_kv_heads, head_dim, causal, QKVLayout::kNHD, rotary_mode);

      thrust::device_vector<int32_t> q_indptr_device(q_indptr);
      thrust::device_vector<T> q_device(q);
      thrust::device_vector<T> o_device(q_len * num_qo_heads * head_dim);

      for (uint32_t num_runs = 0; num_runs < 10; ++num_runs) {
        auto status = BatchPrefillWithPagedKVCache<T, T, int32_t>(
            thrust::raw_pointer_cast(q_device.data()), paged_kv,
            thrust::raw_pointer_cast(q_indptr_device.data()),
            thrust::raw_pointer_cast(o_device.data()), nullptr, num_qo_heads, causal, rotary_mode);
        EXPECT_EQ(status, cudaSuccess) << "CUDA error: " + std::string(cudaGetErrorString(status));
      }

      thrust::host_vector<T> o_host(o_device);
      size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
      bool nan_detected = false;
      for (size_t i = 0; i < q_len * num_qo_heads * head_dim; ++i) {
        if (std::isnan(float(o_host[i]))) {
          nan_detected = true;
        }
        num_result_errors_atol_1e_3_rtol_1e_3 +=
            (!utils::isclose(float(o_host[i]), float(o_ref[i]), 1e-3, 1e-3));
      }
      float result_accuracy = 1. - float(num_result_errors_atol_1e_3_rtol_1e_3) /
                                       max(float(q_len * num_qo_heads * head_dim), 1.f);
      std::cout << "layer_idx=" << layer_idx << ", request_idx=" << request_idx
                << ", num_kv_heads=" << num_kv_heads << ", q_len=" << q_len << ", kv_len=" << kv_len
                << ", head_dim=" << head_dim << ", causal=" << causal
                << ", rotary_mode=" << RotaryModeToString(rotary_mode)
                << ", result_accuracy=" << result_accuracy << std::endl;
      EXPECT_GT(result_accuracy, 0.99) << "Result correctness test failed.";
      EXPECT_EQ(nan_detected, false) << "NaN detected in output.";
    }
  }
}

template <typename T>
void TestBatchPrefillKernelOneHotCorrectness() {
  for (size_t num_kv_heads : {32}) {
    for (size_t head_dim : {128}) {
      for (size_t causal : {false, true}) {
        for (size_t rotary_mode : {0, 1}) {
          _TestBatchPrefillKernelOneHotCorrectness<T>(num_kv_heads, head_dim, causal,
                                                      RotaryMode(rotary_mode));
        }
      }
    }
  }
}

TEST(FlashInferCorrectnessTest, BatchPrefillKernelCorrectnessTestOneHotFP16) {
  TestBatchPrefillKernelOneHotCorrectness<half>();
}
