// 模块：Graph IR 校验器实现
//
// 职责：
//   构建 ValidationGraph 视图，注入 FusionRuleTable，调用公共 V1~V16 规则。
//
// 边界：
//   - 不修改 Graph；
//   - 不执行 V17 运行时校验。

#include "graph/graph_validator.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "graph/fusion_rule.h"
#include "graph/graph.h"
#include "graph/tensor_desc.h"
#include "graph/validation_common.h"

namespace mobilenet {
namespace graph {
namespace {

// 把 Graph IR 节点转换为校验视图。
ValidationNode MakeValidationNode(const Node& node) {
  ValidationNode view;
  view.name = node.name;
  view.op_type = node.op_type;
  view.precision = ScalarTypeName(node.precision);
  view.kernel_key = node.kernel_key;
  view.fusion_rule_id = node.fusion_rule_id;
  view.binding_name = node.binding_name;
  view.copy_kind = node.copy_kind;
  view.input_edges = node.input_edges;
  view.output_edges = node.output_edges;
  view.fused_ops = node.fused_ops;
  view.attributes = node.attributes;
  view.has_host_tensor = node.host_tensor.has_value();
  if (node.host_tensor.has_value()) {
    view.host_tensor = DescribeTensor(*node.host_tensor,
                                      ConcreteSymbolicShape(*node.host_tensor));
  }
  return view;
}

// 把 Graph IR 边转换为校验视图。
ValidationEdge MakeValidationEdge(const Edge& edge,
                                  const std::vector<Node>& nodes) {
  ValidationEdge view;
  view.name = edge.name;
  view.tensor =
      DescribeTensor(edge.tensor, edge.symbolic_shape, edge.quantization);
  view.producer = edge.producer < nodes.size() ? nodes[edge.producer].name : "";
  view.consumers.reserve(edge.consumers.size());
  for (NodeId consumer_id : edge.consumers) {
    if (consumer_id < nodes.size()) {
      view.consumers.push_back(nodes[consumer_id].name);
    }
  }
  view.is_weight = edge.is_weight;
  view.weight_ref = edge.weight_ref;
  return view;
}

// 把 Graph IR 权重转换为校验视图。
ValidationWeight MakeValidationWeight(const WeightDesc& weight) {
  ValidationWeight view;
  view.ref = weight.ref;
  view.source_name = weight.source_name;
  view.tensor = DescribeTensor(
      weight.tensor, ConcreteSymbolicShape(weight.tensor), weight.quantization);
  view.quantization = weight.quantization;
  view.checksum = weight.checksum;
  return view;
}

// 把 Graph IR 模型配置转换为校验视图。
ValidationModelConfig MakeValidationConfig(const ModelConfig& config) {
  ValidationModelConfig view;
  view.model_family = config.model_family;
  view.num_layers = config.num_layers;
  view.hidden_size = config.hidden_size;
  view.intermediate_size = config.intermediate_size;
  view.num_attention_heads = config.num_attention_heads;
  view.num_kv_heads = config.num_kv_heads;
  view.head_dim = config.head_dim;
  view.vocab_size = config.vocab_size;
  view.max_context_length = config.max_context_length;
  view.norm_eps = config.norm_eps;
  view.activation = config.activation;
  view.tie_word_embeddings = config.tie_word_embeddings;
  view.dtype = ScalarTypeName(config.dtype);
  view.rope = config.rope;
  view.extra = config.extra;
  return view;
}

// 构造规则视图映射，保证 Lookup 返回的指针在本次校验期间稳定。
std::map<std::string, RuleView> MakeRuleViews() {
  std::map<std::string, RuleView> views;
  for (const FusionRule& rule : FusionRuleTable::Instance().rules()) {
    RuleView view;
    view.rule_id = rule.rule_id;
    view.model_family = rule.model_family;
    view.stage = rule.stage;
    view.op_type = rule.op_type;
    view.required_op_kinds = rule.required_op_kinds;
    view.required_attributes = rule.required_attributes;
    view.kernel_key = rule.kernel_key;
    views.emplace(rule.rule_id, std::move(view));
  }
  return views;
}

}  // namespace

Status GraphValidator::Validate(const Graph& graph, ValidationReport* report) {
  if (report == nullptr) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "validation report must not be null");
  }
  ValidationGraph view;
  view.schema_major = graph.schema_version_major();
  view.schema_minor = graph.schema_version_minor();
  view.fusion_rules_version = graph.fusion_rules_version();
  view.model_config = MakeValidationConfig(graph.model_config());
  view.nodes.reserve(graph.nodes().size());
  for (const Node& node : graph.nodes()) {
    view.nodes.push_back(MakeValidationNode(node));
  }
  view.edges.reserve(graph.edges().size());
  for (const Edge& edge : graph.edges()) {
    view.edges.push_back(MakeValidationEdge(edge, graph.nodes()));
  }
  view.weights.reserve(graph.weights().size());
  for (const WeightDesc& weight : graph.weights()) {
    view.weights.push_back(MakeValidationWeight(weight));
  }
  view.topo_order = graph.topo_order();
  view.input_nodes = graph.input_nodes();
  view.output_nodes = graph.output_nodes();

  const std::map<std::string, RuleView> rule_views = MakeRuleViews();
  const RuleLookup lookup =
      [&rule_views](std::string_view rule_id) -> const RuleView* {
    const auto iterator = rule_views.find(std::string(rule_id));
    return iterator == rule_views.end() ? nullptr : &iterator->second;
  };
  return ValidateGraphView(view, lookup, report);
}

}  // namespace graph
}  // namespace mobilenet
