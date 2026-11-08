// 模块：RuntimeGraph 图对象实现
//
// 职责：
//   实现运行时图查询、V1~V16 校验入口与规范 JSON 输出。
//
// 边界：
//   - 不执行 Forward、不管理 Kernel/Arena 生命周期（见 forward 源文件）；
//   - 不访问张量数据。

#include "graph/runtime_graph.h"

#include <cassert>
#include <string>
#include <utility>

#include "graph/canonical_json.h"
#include "graph/json_writer.h"
#include "graph/runtime_validator.h"

namespace mobilenet {
namespace graph {
namespace {

// TensorDesc -> 规范 JSON 视图。
ValidationTensor MakeValidationTensor(const TensorDesc& tensor) {
  ValidationTensor view;
  view.defined = true;
  view.dtype = DataTypeName(tensor.dtype);
  view.shape = tensor.shape;
  view.stride = tensor.stride;
  view.layout = tensor.layout;
  view.device = tensor.device;
  view.offset = tensor.offset;
  view.size_bytes = tensor.size_bytes;
  view.quantization = tensor.quantization;
  return view;
}

// RuntimeModelConfig -> 校验/JSON 视图。
ValidationModelConfig MakeValidationConfig(const RuntimeModelConfig& config) {
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
  view.dtype = DataTypeName(config.dtype);
  view.rope = config.rope;
  view.extra = config.extra;
  return view;
}

// 写入节点名称数组。
void WriteNodeNames(const std::vector<NodeId>& node_ids,
                    const std::vector<RuntimeNode>& nodes, JsonWriter* writer) {
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

// 写入边名称数组。
void WriteEdgeNames(const std::vector<EdgeId>& edge_ids,
                    const std::vector<RuntimeEdge>& edges, JsonWriter* writer) {
  writer->BeginArray();
  for (EdgeId edge_id : edge_ids) {
    if (edge_id < edges.size()) {
      writer->String(edges[edge_id].name);
    } else {
      writer->String("");
    }
  }
  writer->EndArray();
}

}  // namespace

RuntimeGraph::~RuntimeGraph() {
  // 仅调试构建检查“绑定过 Kernel 必须先 Release”，Release 构建不产生开销。
#ifndef NDEBUG
  bool has_kernel = false;
  for (Kernel* kernel : node_kernels_) {
    if (kernel != nullptr) {
      has_kernel = true;
      break;
    }
  }
  assert(released_ || !has_kernel);
#endif
}

NodeId RuntimeGraph::FindNode(std::string_view name) const {
  const auto iterator = node_ids_.find(std::string(name));
  return iterator == node_ids_.end() ? kInvalidNodeId : iterator->second;
}

EdgeId RuntimeGraph::FindEdge(std::string_view name) const {
  const auto iterator = edge_ids_.find(std::string(name));
  return iterator == edge_ids_.end() ? kInvalidEdgeId : iterator->second;
}

const RuntimeNode* RuntimeGraph::GetNode(NodeId id) const {
  return id < nodes_.size() ? &nodes_[id] : nullptr;
}

const RuntimeEdge* RuntimeGraph::GetEdge(EdgeId id) const {
  return id < edges_.size() ? &edges_[id] : nullptr;
}

const RuntimeWeightDesc* RuntimeGraph::GetWeight(std::string_view ref) const {
  const auto iterator = weight_indices_.find(std::string(ref));
  return iterator == weight_indices_.end() ? nullptr
                                           : &weights_[iterator->second];
}

Status RuntimeGraph::Validate(ValidationReport* report) const {
  return RuntimeGraphValidator::Validate(*this, report);
}

std::string RuntimeGraph::ToCanonicalJson() const {
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
  for (const RuntimeNode& node : nodes_) {
    writer.BeginObject();
    writer.Key("name");
    writer.String(node.name);
    writer.Key("op_type");
    writer.String(OpTypeName(node.op_type));
    writer.Key("precision");
    writer.String(DataTypeName(node.precision));
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
    WriteEdgeNames(node.input_edges, edges_, &writer);
    writer.Key("output_edges");
    WriteEdgeNames(node.output_edges, edges_, &writer);
    writer.Key("host_tensor");
    if (node.host_tensor.has_value()) {
      WriteTensorJson(MakeValidationTensor(*node.host_tensor), &writer);
    } else {
      writer.Null();
    }
    writer.EndObject();
  }
  writer.EndArray();

  writer.Key("edges");
  writer.BeginArray();
  for (const RuntimeEdge& edge : edges_) {
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
    WriteTensorJson(MakeValidationTensor(edge.tensor), &writer);
    writer.EndObject();
  }
  writer.EndArray();

  writer.Key("weights");
  writer.BeginArray();
  for (const RuntimeWeightDesc& weight : weights_) {
    writer.BeginObject();
    writer.Key("ref");
    writer.String(weight.ref);
    writer.Key("source_name");
    writer.String(weight.source_name);
    writer.Key("checksum");
    writer.UInt(weight.checksum);
    writer.Key("tensor");
    WriteTensorJson(MakeValidationTensor(weight.tensor), &writer);
    writer.EndObject();
  }
  writer.EndArray();

  writer.EndObject();
  return writer.Take();
}

}  // namespace graph
}  // namespace mobilenet
