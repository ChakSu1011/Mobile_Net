// 模块：Graph 公共校验规则 V1~V16
//
// 职责：
//   用与张量表示无关的 ValidationGraph 视图实现 Graph IR 与 RuntimeGraph
//   共用的结构校验规则，避免两套校验逻辑漂移。
//
// 边界：
//   - 不包含 at::Tensor、TensorDesc 的构造/转换逻辑；
//   - V17（运行时执行前校验）不在此文件，见 runtime_validator.h；
//   - V13 的融合规则查询通过 RuleLookup 注入：Graph IR 注入完整规则表，
//     RuntimeGraph 可选择跳过规则存在性检查，只做结构检查。
//
// 依赖：
//   graph_common.h、Runtime/kernel/kernel_types.h。

#ifndef MOBILENET_GRAPH_CPP_COMMON_INCLUDE_GRAPH_VALIDATION_COMMON_H_
#define MOBILENET_GRAPH_CPP_COMMON_INCLUDE_GRAPH_VALIDATION_COMMON_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "graph/graph_common.h"

namespace mobilenet {
namespace graph {

// 校验视图中的张量元数据。
//
// 说明：
//   - Graph IR 由 at::Tensor + SymbolicShape 填充；
//   - RuntimeGraph 由 TensorDesc 填充；
//   - storage_key 仅 Graph IR 填写（用于 V8 权重共享检查），RuntimeGraph 留空。
struct ValidationTensor {
  bool defined = false;
  std::string dtype;
  std::vector<Dim> shape;
  std::vector<int64_t> stride;
  Layout layout = Layout::kRowMajor;
  DeviceType device = DeviceType::kHost;
  uint64_t offset = 0;
  uint64_t size_bytes = 0;
  bool requires_grad = false;
  const void* storage_key = nullptr;
  std::optional<QuantizationDesc> quantization;
};

// 校验视图中的节点。
struct ValidationNode {
  std::string name;
  OpType op_type = OpType::kUnknown;
  std::string precision;
  std::string kernel_key;
  std::string fusion_rule_id;
  std::string binding_name;
  CopyKind copy_kind = CopyKind::kNone;
  std::vector<std::string> input_edges;
  std::vector<std::string> output_edges;
  std::vector<FusedOp> fused_ops;
  std::map<std::string, AttributeValue> attributes;
  bool has_host_tensor = false;
  ValidationTensor host_tensor;
};

// 校验视图中的边。
struct ValidationEdge {
  std::string name;
  ValidationTensor tensor;
  std::string producer;
  std::vector<std::string> consumers;
  bool is_weight = false;
  std::string weight_ref;
};

// 校验视图中的权重。
struct ValidationWeight {
  std::string ref;
  std::string source_name;
  ValidationTensor tensor;
  std::optional<QuantizationDesc> quantization;
  uint64_t checksum = 0;
};

// 校验视图中的模型配置。
struct ValidationModelConfig {
  std::string model_family;
  int32_t num_layers = 0;
  int32_t hidden_size = 0;
  int32_t intermediate_size = 0;
  int32_t num_attention_heads = 0;
  int32_t num_kv_heads = 0;
  int32_t head_dim = 0;
  int32_t vocab_size = 0;
  int32_t max_context_length = 0;
  double norm_eps = 0.0;
  std::string activation;
  bool tie_word_embeddings = false;
  std::string dtype;
  RopeConfig rope;
  std::map<std::string, AttributeValue> extra;
};

// 校验视图。
struct ValidationGraph {
  uint32_t schema_major = 1;
  uint32_t schema_minor = 0;
  std::string fusion_rules_version;
  ValidationModelConfig model_config;
  std::vector<ValidationNode> nodes;
  std::vector<ValidationEdge> edges;
  std::vector<ValidationWeight> weights;
  std::vector<NodeId> topo_order;
  std::vector<NodeId> input_nodes;
  std::vector<NodeId> output_nodes;
};

// 融合规则的最小只读视图。
struct RuleView {
  std::string rule_id;
  std::string model_family;
  BlockStage stage = BlockStage::kUnknown;
  OpType op_type = OpType::kUnknown;
  std::vector<std::string> required_op_kinds;
  std::map<std::string, AttributeValue> required_attributes;
  std::string kernel_key;
};

// 融合规则查询函数；返回 nullptr 表示未找到。
using RuleLookup = std::function<const RuleView*(std::string_view rule_id)>;

// 执行 V1~V16 公共校验。
//
// 参数：
//   graph：校验视图；
//   rule_lookup：融合规则查询；为空时 V13 只检查规则 ID/算子序列非空，
//                不检查规则是否存在于规则表。
//   report：输出报告，函数开始时清空。
// 返回：
//   校验通过返回 Status::Ok()；否则返回首个问题的错误码。
Status ValidateGraphView(const ValidationGraph& graph,
                         const RuleLookup& rule_lookup,
                         ValidationReport* report);

// 将稳定 dtype 名称解析为 DataType；未知返回 kUnknown。
DataType DataTypeFromName(std::string_view name);

}  // namespace graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_CPP_COMMON_INCLUDE_GRAPH_VALIDATION_COMMON_H_
