// 模块：Graph IR 图对象实现
//
// 职责：
//   实现节点/边/权重查询、节点输入输出张量视图、校验入口与规范 JSON。
//
// 边界：
//   - 不访问张量数据；
//   - 不执行图变换或运行时绑定。

#include "graph/graph.h"

#include <string>
#include <utility>

#include "graph/canonical_json.h"
#include "graph/graph_validator.h"
#include "graph/json_writer.h"
#include "graph/tensor_desc.h"

namespace mobilenet {
namespace graph {
namespace {

// 把 Graph IR 模型配置转换为校验/JSON 视图。
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

// 写入节点名称数组。
void WriteNodeNames(const std::vector<NodeId>& node_ids,
                    const std::vector<Node>& nodes, JsonWriter* writer) {
  writer->BeginArray();
  for (NodeId node_id : node_ids) {
    if (node_id < nodes.size()) {
      writer->String(nodes[node_id].name);
    } else {
      writer->String("");
    }
  }
  writer->EndArray();
}

}  // namespace

NodeId Graph::FindNode(std::string_view name) const {
  const auto iterator = node_ids_.find(std::string(name));
  return iterator == node_ids_.end() ? kInvalidNodeId : iterator->second;
}

EdgeId Graph::FindEdge(std::string_view name) const {
  const auto iterator = edge_ids_.find(std::string(name));
  return iterator == edge_ids_.end() ? kInvalidEdgeId : iterator->second;
}

const WeightDesc* Graph::FindWeight(std::string_view ref) const {
  const auto iterator = weight_indices_.find(std::string(ref));
  return iterator == weight_indices_.end() ? nullptr
                                           : &weights_[iterator->second];
}

const Node* Graph::GetNode(NodeId id) const {
  return id < nodes_.size() ? &nodes_[id] : nullptr;
}

const Edge* Graph::GetEdge(EdgeId id) const {
  return id < edges_.size() ? &edges_[id] : nullptr;
}

std::vector<at::Tensor> Graph::NodeInputs(NodeId node_id) const {
  std::vector<at::Tensor> tensors;
  const Node* node = GetNode(node_id);
  if (node == nullptr) {
    return tensors;
  }
  tensors.reserve(node->input_edges.size());
  for (const std::string& edge_name : node->input_edges) {
    const EdgeId edge_id = FindEdge(edge_name);
    if (edge_id != kInvalidEdgeId) {
      tensors.push_back(edges_[edge_id].tensor);
    }
  }
  return tensors;
}

std::vector<at::Tensor> Graph::NodeOutputs(NodeId node_id) const {
  std::vector<at::Tensor> tensors;
  const Node* node = GetNode(node_id);
  if (node == nullptr) {
    return tensors;
  }
  tensors.reserve(node->output_edges.size());
  for (const std::string& edge_name : node->output_edges) {
    const EdgeId edge_id = FindEdge(edge_name);
    if (edge_id != kInvalidEdgeId) {
      tensors.push_back(edges_[edge_id].tensor);
    }
  }
  return tensors;
}

Status Graph::Validate(ValidationReport* report) const {
  ValidationReport local_report;
  ValidationReport* target = report != nullptr ? report : &local_report;
  return GraphValidator::Validate(*this, target);
}

std::string Graph::ToCanonicalJson() const {
  JsonWriter writer;
  writer.BeginObject();
  writer.Key("schema_version_major");
  writer.Int(schema_version_major_);
  writer.Key("schema_version_minor");
  writer.Int(schema_version_minor_);
  writer.Key("fusion_rules_version");
  writer.String(fusion_rules_version_);
  writer.Key("model_config");
  WriteModelConfigJson(MakeValidationConfig(model_config_), &writer);

  writer.Key("input_nodes");
  WriteNodeNames(input_nodes_, nodes_, &writer);
  writer.Key("output_nodes");
  WriteNodeNames(output_nodes_, nodes_, &writer);

  writer.Key("topo_order");
  writer.BeginArray();
  for (NodeId node_id : topo_order_) {
    if (node_id < nodes_.size()) {
      writer.String(nodes_[node_id].name);
    } else {
      writer.String("");
    }
  }
  writer.EndArray();

  writer.Key("nodes");
  writer.BeginArray();
  for (const Node& node : nodes_) {
    writer.BeginObject();
    writer.Key("name");
    writer.String(node.name);
    writer.Key("op_type");
    writer.String(OpTypeName(node.op_type));
    writer.Key("precision");
    writer.String(ScalarTypeName(node.precision));
    writer.Key("kernel_key");
    writer.String(node.kernel_key);
    writer.Key("fusion_rule_id");
    writer.String(node.fusion_rule_id);
    writer.Key("binding_name");
    writer.String(node.binding_name);
    writer.Key("copy_kind");
    writer.String(CopyKindName(node.copy_kind));
    writer.Key("fused_ops");
    WriteFusedOpsJson(node.fused_ops, &writer);
    writer.Key("attributes");
    WriteAttributesJson(node.attributes, &writer);
    writer.Key("input_edges");
    writer.BeginArray();
    for (const std::string& edge_name : node.input_edges) {
      writer.String(edge_name);
    }
    writer.EndArray();
    writer.Key("output_edges");
    writer.BeginArray();
    for (const std::string& edge_name : node.output_edges) {
      writer.String(edge_name);
    }
    writer.EndArray();
    writer.Key("host_tensor");
    if (node.host_tensor.has_value()) {
      WriteTensorJson(DescribeTensor(*node.host_tensor,
                                     ConcreteSymbolicShape(*node.host_tensor),
                                     std::nullopt),
                      &writer);
    } else {
      writer.Null();
    }
    writer.EndObject();
  }
  writer.EndArray();

  writer.Key("edges");
  writer.BeginArray();
  for (const Edge& edge : edges_) {
    writer.BeginObject();
    writer.Key("name");
    writer.String(edge.name);
    writer.Key("producer");
    if (edge.producer < nodes_.size()) {
      writer.String(nodes_[edge.producer].name);
    } else {
      writer.String("");
    }
    writer.Key("consumers");
    WriteNodeNames(edge.consumers, nodes_, &writer);
    writer.Key("is_weight");
    writer.Bool(edge.is_weight);
    writer.Key("weight_ref");
    writer.String(edge.weight_ref);
    writer.Key("tensor");
    WriteTensorJson(
        DescribeTensor(edge.tensor, edge.symbolic_shape, edge.quantization),
        &writer);
    writer.EndObject();
  }
  writer.EndArray();

  writer.Key("weights");
  writer.BeginArray();
  for (const WeightDesc& weight : weights_) {
    writer.BeginObject();
    writer.Key("ref");
    writer.String(weight.ref);
    writer.Key("source_name");
    writer.String(weight.source_name);
    writer.Key("checksum");
    writer.UInt(weight.checksum);
    writer.Key("tensor");
    WriteTensorJson(
        DescribeTensor(weight.tensor, ConcreteSymbolicShape(weight.tensor),
                       weight.quantization),
        &writer);
    writer.EndObject();
  }
  writer.EndArray();

  writer.EndObject();
  return writer.Take();
}

}  // namespace graph
}  // namespace mobilenet
