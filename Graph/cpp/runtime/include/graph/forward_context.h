// 模块：RuntimeGraph Forward 上下文
//
// 职责：
//   定义一次 Forward 所需的 I/O 绑定、KV Cache 视图、Attention 元数据、
//   Sampling 参数、workspace、Backend/Stream，以及传给 Kernel 的
//   KernelContext。
//
// 边界：
//   - 所有字段使用裸指针/POD/标准容器，不包含 LibTorch；
//   - KernelContext 由 RuntimeGraph 在栈上构造，生命周期仅限一次 Execute；
//   - 本文件不分配设备内存、不实现 Kernel。
//
// 依赖：
//   runtime_graph_types.h、Runtime/kernel/kernel.h、
//   Runtime/backend/backend.h。

#ifndef MOBILENET_GRAPH_CPP_RUNTIME_INCLUDE_GRAPH_FORWARD_CONTEXT_H_
#define MOBILENET_GRAPH_CPP_RUNTIME_INCLUDE_GRAPH_FORWARD_CONTEXT_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "Runtime/backend/backend.h"
#include "Runtime/kernel/kernel.h"
#include "graph/runtime_graph_types.h"

namespace mobilenet {
namespace graph {

// Arena 绑定。
struct ArenaBinding {
  void* base = nullptr;
  uint64_t bytes = 0;
};

// 主机 I/O 绑定。
//
// 字段：
//   host_ptr：主机缓冲区地址；
//   bytes：缓冲区字节数；
//   shape：本次调用的具体形状（符号维取绑定值）；
//   dtype：主机数据类型；
//   device：必须为 kHost。
struct IoBinding {
  void* host_ptr = nullptr;
  uint64_t bytes = 0;
  std::vector<Dim> shape;
  DataType dtype = DataType::kUnknown;
  DeviceType device = DeviceType::kHost;
};

// KV Cache 视图。
//
// 字段：
//   base：KV Cache 设备内存基址；
//   bytes：总字节数；
//   num_layers/num_kv_heads/head_dim：结构参数；
//   block_size/max_blocks：分页参数；
//   dtype：cache 数据类型。
struct KVCacheView {
  void* base = nullptr;
  uint64_t bytes = 0;
  int32_t num_layers = 0;
  int32_t num_kv_heads = 0;
  int32_t head_dim = 0;
  int32_t block_size = 0;
  int32_t max_blocks = 0;
  DataType dtype = DataType::kUnknown;
};

// Attention 元数据。
struct AttentionMetadata {
  std::vector<int32_t> block_tables;
  std::vector<int32_t> sequence_lengths;
  std::vector<int32_t> slot_mapping;
  int32_t batch_size = 0;
  int32_t max_context_length = 0;
  bool is_prefill = true;
};

// 单条 Sampling 参数。
struct SamplingParams {
  double temperature = 1.0;
  double top_p = 1.0;
  int32_t top_k = 0;
  double repetition_penalty = 1.0;
  int32_t max_new_tokens = 1;
  uint64_t seed = 0;
  bool greedy = false;
};

// 批量 Sampling 参数。
struct SamplingBatch {
  std::vector<SamplingParams> params;
  int32_t batch_size = 0;
};

// Forward 上下文。
struct ForwardContext {
  std::map<std::string, IoBinding> host_inputs;
  std::map<std::string, IoBinding> host_outputs;
  KVCacheView kv_cache;
  AttentionMetadata attention;
  SamplingBatch sampling;
  void* workspace = nullptr;
  uint64_t workspace_bytes = 0;
  Backend* backend = nullptr;
  Stream* stream = nullptr;
};

// Kernel 执行上下文。
//
// 约束：
//   - input_tensors/output_tensors/input_ptrs/output_ptrs 指向 RuntimeGraph
//     Forward 栈上的数组，Kernel 不得保存；
//   - attributes 指向 RuntimeNode 的只读属性表。
struct KernelContext {
  const RuntimeNode* node = nullptr;
  const TensorDesc* const* input_tensors = nullptr;
  const TensorDesc* const* output_tensors = nullptr;
  int32_t num_inputs = 0;
  int32_t num_outputs = 0;
  void* const* input_ptrs = nullptr;
  void* const* output_ptrs = nullptr;
  const std::map<std::string, AttributeValue>* attributes = nullptr;
  const KVCacheView* kv_cache = nullptr;
  const AttentionMetadata* attention = nullptr;
  const SamplingBatch* sampling = nullptr;
  void* workspace = nullptr;
  uint64_t workspace_bytes = 0;
  Stream* stream = nullptr;
};

}  // namespace graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_CPP_RUNTIME_INCLUDE_GRAPH_FORWARD_CONTEXT_H_
