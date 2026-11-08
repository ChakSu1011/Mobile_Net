// 模块：RuntimeGraph 数据类型
//
// 职责：
//   定义运行时侧 TensorDesc、RuntimeNode、RuntimeEdge、RuntimeWeightDesc 与
//   模型配置。所有张量只保留描述信息与裸偏移，不包含 at::Tensor/LibTorch。
//
// 边界：
//   - 本文件不得包含任何 ATen/torch/c10 头文件；
//   - TensorDesc 不拥有内存，offset 相对于所属 arena 基址；
//   - 不定义 Kernel/Backend/Forward 接口。

#ifndef MOBILENET_GRAPH_CPP_RUNTIME_INCLUDE_GRAPH_RUNTIME_GRAPH_TYPES_H_
#define MOBILENET_GRAPH_CPP_RUNTIME_INCLUDE_GRAPH_RUNTIME_GRAPH_TYPES_H_

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "graph/graph_common.h"

namespace mobilenet {
namespace graph {

// 运行时张量描述。
//
// 字段：
//   dtype：元素类型；
//   shape：形状（可含符号维，符号维 value 为当前绑定值）；
//   stride：行/列主序步长，单位元素；
//   layout：行主序或列主序；
//   device：Host 或 Device；
//   offset：相对 arena 基址或主机缓冲区的字节偏移；
//   size_bytes：张量载荷字节数；
//   quantization：可选量化描述。
struct TensorDesc {
  DataType dtype = DataType::kUnknown;
  std::vector<Dim> shape;
  std::vector<int64_t> stride;
  Layout layout = Layout::kRowMajor;
  DeviceType device = DeviceType::kHost;
  uint64_t offset = 0;
  uint64_t size_bytes = 0;
  std::optional<QuantizationDesc> quantization;

  // 秩。
  size_t Rank() const { return shape.size(); }
};

// RuntimeGraph 的边。
//
// 字段：
//   producer：产生该边的节点 ID；权重边为 kInvalidNodeId；
//   consumers：消费该边的节点 ID；
//   arena_id/memory_offset：Loader/MemoryPlan 绑定的设备内存位置。
struct RuntimeEdge {
  std::string name;
  TensorDesc tensor;
  std::optional<QuantizationDesc> quantization;
  NodeId producer = kInvalidNodeId;
  std::vector<NodeId> consumers;
  bool is_weight = false;
  std::string weight_ref;
  int32_t arena_id = -1;
  uint64_t memory_offset = 0;
};

// RuntimeGraph 的权重描述。
//
// 字段：
//   tensor：权重张量描述；
//   data_offset/data_bytes：权重字节流在权重 arena 中的位置；
//   checksum：序列化校验和，0 表示未计算。
struct RuntimeWeightDesc {
  std::string ref;
  std::string source_name;
  TensorDesc tensor;
  uint64_t data_offset = 0;
  uint64_t data_bytes = 0;
  uint64_t checksum = 0;
};

// RuntimeGraph 的节点。
struct RuntimeNode {
  std::string name;
  OpType op_type = OpType::kUnknown;
  std::vector<FusedOp> fused_ops;
  std::map<std::string, AttributeValue> attributes;
  std::vector<EdgeId> input_edges;
  std::vector<EdgeId> output_edges;
  DataType precision = DataType::kUnknown;
  std::string kernel_key;
  std::string fusion_rule_id;
  std::string binding_name;
  CopyKind copy_kind = CopyKind::kNone;
  std::optional<TensorDesc> host_tensor;
};

// RuntimeGraph 模型配置；dtype 使用 DataType。
using RuntimeModelConfig = ModelConfigT<DataType>;

}  // namespace graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_CPP_RUNTIME_INCLUDE_GRAPH_RUNTIME_GRAPH_TYPES_H_
