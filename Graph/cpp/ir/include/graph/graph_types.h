// 模块：Graph IR 数据类型
//
// 职责：
//   定义 Graph IR 的节点、边、权重与模型配置类型。所有张量属性直接使用
//   at::Tensor，精度直接使用 at::ScalarType，不自定义底层张量元素。
//
// 边界：
//   - 本文件属于工具链侧，允许包含 ATen；RuntimeGraph 不得包含本文件；
//   - 不包含图构建、校验、序列化或 Python 绑定逻辑。
//
// 依赖：
//   ATen（at::Tensor / at::ScalarType）、graph_common.h。

#ifndef MOBILENET_GRAPH_CPP_IR_INCLUDE_GRAPH_GRAPH_TYPES_H_
#define MOBILENET_GRAPH_CPP_IR_INCLUDE_GRAPH_GRAPH_TYPES_H_

#include <ATen/ATen.h>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "graph/graph_common.h"

namespace mobilenet {
namespace graph {

// Graph IR 模型配置；dtype 使用 at::ScalarType。
using ModelConfig = ModelConfigT<at::ScalarType>;

// Graph IR 的边。
//
// 功能：
//   表示一条数据流边。激活边由 producer 节点产生、consumers 节点消费；
//   权重边没有 producer，通过 weight_ref 指向 WeightDesc。
//
// 边界：
//   - tensor 必须已定义；权重边必须与 WeightDesc 共享存储；
//   - symbolic_shape 提供符号维元数据，tensor 提供具体示例形状；
//   - quantization 只描述存储量化，不改变 tensor 的 dtype。
struct Edge {
  std::string name;
  at::Tensor tensor;
  SymbolicShape symbolic_shape;
  std::optional<QuantizationDesc> quantization;
  NodeId producer = kInvalidNodeId;
  std::vector<NodeId> consumers;
  bool is_weight = false;
  std::string weight_ref;

  // 构造边。
  //
  // 参数：
  //   edge_name：边名，必须符合图名称规则；
  //   edge_tensor：已定义的 at::Tensor；
  //   edge_shape：符号形状，必须与 tensor 的 sizes 一致；
  //   edge_quantization：可选量化描述；
  //   edge_is_weight：是否为权重边；
  //   edge_weight_ref：权重边的 WeightDesc.ref。
  Edge(std::string edge_name, at::Tensor edge_tensor, SymbolicShape edge_shape,
       std::optional<QuantizationDesc> edge_quantization = std::nullopt,
       bool edge_is_weight = false, std::string edge_weight_ref = "")
      : name(std::move(edge_name)),
        tensor(std::move(edge_tensor)),
        symbolic_shape(std::move(edge_shape)),
        quantization(std::move(edge_quantization)),
        is_weight(edge_is_weight),
        weight_ref(std::move(edge_weight_ref)) {}
};

// Graph IR 的权重描述。
//
// 功能：
//   tensor 保存模型参数的 at::Tensor（CPU 或 CUDA）；source_name 为原始
//   参数名，便于调试与序列化映射。
struct WeightDesc {
  std::string ref;
  std::string source_name;
  at::Tensor tensor;
  std::optional<QuantizationDesc> quantization;
  uint64_t checksum = 0;

  // 构造权重描述。
  WeightDesc(std::string weight_ref, std::string weight_source_name,
             at::Tensor weight_tensor,
             std::optional<QuantizationDesc> weight_quantization = std::nullopt,
             uint64_t weight_checksum = 0)
      : ref(std::move(weight_ref)),
        source_name(std::move(weight_source_name)),
        tensor(std::move(weight_tensor)),
        quantization(std::move(weight_quantization)),
        checksum(weight_checksum) {}
};

// Graph IR 的节点。
//
// 功能：
//   表示一个融合算子节点；I/O 节点通过 binding_name/copy_kind/host_tensor
//   描述图边界，计算节点通过 fused_ops/kernel_key/fusion_rule_id 描述实现。
//
// 边界：
//   - 不拥有输入/输出边的存储，只保存边名；
//   - host_tensor 只用于 I/O 节点；计算节点通常为空。
struct Node {
  std::string name;
  OpType op_type = OpType::kUnknown;
  std::vector<FusedOp> fused_ops;
  std::map<std::string, AttributeValue> attributes;
  std::vector<std::string> input_edges;
  std::vector<std::string> output_edges;
  at::ScalarType precision = at::ScalarType::Undefined;
  std::string kernel_key;
  std::string fusion_rule_id;
  std::string binding_name;
  CopyKind copy_kind = CopyKind::kNone;
  std::optional<at::Tensor> host_tensor;

  // 默认构造；Python 绑定与 Builder 使用显式字段构造。
  Node() = default;

  // 构造节点。
  //
  // 参数：
  //   node_name：节点名；
  //   node_op_type：算子类型；
  //   node_fused_ops：融合前的原始算子序列；
  //   node_attributes：节点属性；
  //   node_input_edges：输入边名；
  //   node_output_edges：输出边名；
  //   node_precision：计算精度；
  //   node_kernel_key：后端实现键；
  //   node_fusion_rule_id：融合规则 ID；
  //   node_binding_name：I/O 绑定名；
  //   node_copy_kind：I/O 拷贝方向；
  //   node_host_tensor：I/O 主机张量。
  Node(std::string node_name, OpType node_op_type,
       std::vector<FusedOp> node_fused_ops,
       std::map<std::string, AttributeValue> node_attributes,
       std::vector<std::string> node_input_edges,
       std::vector<std::string> node_output_edges,
       at::ScalarType node_precision, std::string node_kernel_key = "",
       std::string node_fusion_rule_id = "", std::string node_binding_name = "",
       CopyKind node_copy_kind = CopyKind::kNone,
       std::optional<at::Tensor> node_host_tensor = std::nullopt)
      : name(std::move(node_name)),
        op_type(node_op_type),
        fused_ops(std::move(node_fused_ops)),
        attributes(std::move(node_attributes)),
        input_edges(std::move(node_input_edges)),
        output_edges(std::move(node_output_edges)),
        precision(node_precision),
        kernel_key(std::move(node_kernel_key)),
        fusion_rule_id(std::move(node_fusion_rule_id)),
        binding_name(std::move(node_binding_name)),
        copy_kind(node_copy_kind),
        host_tensor(std::move(node_host_tensor)) {}
};

}  // namespace graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_CPP_IR_INCLUDE_GRAPH_GRAPH_TYPES_H_
