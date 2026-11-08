// 模块：RuntimeGraph 构建器实现
//
// 职责：
//   实现运行时图增量收集、边引用解析、拓扑排序与 Finalize 校验。
//
// 边界：
//   - 不分配 arena、不创建 Kernel；
//   - Finalize 失败时不修改 Builder 集合。

#include "graph/runtime_graph_builder.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <set>
#include <utility>

namespace mobilenet {
namespace graph {
namespace {

// 检查 TensorDesc 的基本一致性。
Status CheckTensorDesc(const TensorDesc& tensor, std::string_view context) {
  if (tensor.dtype == DataType::kUnknown) {
    return Status::Error(ErrorCode::kUnsupportedDtype,
                         std::string(context) + ": dtype is unknown");
  }
  if (tensor.shape.size() != tensor.stride.size()) {
    return Status::Error(
        ErrorCode::kInvalidArgument,
        std::string(context) + ": shape rank does not match stride rank");
  }
  if (tensor.layout == Layout::kColumnMajor && tensor.shape.size() != 2) {
    return Status::Error(ErrorCode::kUnsupportedLayout,
                         std::string(context) + ": column-major must be 2-D");
  }
  if (tensor.device == DeviceType::kUnknown) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         std::string(context) + ": device is unknown");
  }
  for (const Dim& dim : tensor.shape) {
    if (dim.value < 0) {
      return Status::Error(
          ErrorCode::kInvalidArgument,
          std::string(context) + ": tensor dimension is negative");
    }
    if (dim.is_symbolic && dim.symbol.empty()) {
      return Status::Error(
          ErrorCode::kInvalidArgument,
          std::string(context) + ": symbolic dimension has empty symbol");
    }
    if (!dim.is_symbolic && !dim.symbol.empty()) {
      return Status::Error(
          ErrorCode::kInvalidArgument,
          std::string(context) + ": constant dimension has non-empty symbol");
    }
  }
  for (int64_t stride : tensor.stride) {
    if (stride < 0) {
      return Status::Error(ErrorCode::kInvalidArgument,
                           std::string(context) + ": stride is negative");
    }
  }
  return Status::Ok();
}

}  // namespace

RuntimeGraphBuilder::RuntimeGraphBuilder() : fusion_rules_version_("1.0") {}

Status RuntimeGraphBuilder::Fail(ErrorCode code, std::string message) {
  validation_report_.Clear();
  validation_report_.AddIssue(ValidationIssue{code, message, "", "", "", ""});
  return Status::Error(code, std::move(message));
}

Status RuntimeGraphBuilder::AddNode(RuntimeNode node, NodeId* node_id) {
  if (finalized_) {
    return Fail(ErrorCode::kStateError,
                "runtime graph builder has already been finalized");
  }
  if (!IsValidGraphName(node.name)) {
    return Fail(ErrorCode::kInvalidArgument,
                "node name is empty or invalid: " + node.name);
  }
  if (node_ids_.find(node.name) != node_ids_.end()) {
    return Fail(ErrorCode::kInvalidArgument,
                "duplicate node name: " + node.name);
  }
  const NodeId new_id = static_cast<NodeId>(nodes_.size());
  nodes_.push_back(std::move(node));
  node_ids_.emplace(nodes_.back().name, new_id);
  if (node_id != nullptr) {
    *node_id = new_id;
  }
  return Status::Ok();
}

Status RuntimeGraphBuilder::AddEdge(RuntimeEdge edge, EdgeId* edge_id) {
  if (finalized_) {
    return Fail(ErrorCode::kStateError,
                "runtime graph builder has already been finalized");
  }
  if (!IsValidGraphName(edge.name)) {
    return Fail(ErrorCode::kInvalidArgument,
                "edge name is empty or invalid: " + edge.name);
  }
  if (edge_ids_.find(edge.name) != edge_ids_.end()) {
    return Fail(ErrorCode::kInvalidArgument,
                "duplicate edge name: " + edge.name);
  }
  Status status = CheckTensorDesc(edge.tensor, "edge " + edge.name);
  if (!status.ok()) {
    return Fail(status.code(), status.message());
  }
  if (edge.is_weight && edge.weight_ref.empty()) {
    return Fail(ErrorCode::kInvalidArgument,
                "weight edge has empty weight_ref: " + edge.name);
  }
  if (!edge.is_weight && !edge.weight_ref.empty()) {
    return Fail(ErrorCode::kInvalidArgument,
                "activation edge has weight_ref: " + edge.name);
  }
  const EdgeId new_id = static_cast<EdgeId>(edges_.size());
  edges_.push_back(std::move(edge));
  edge_ids_.emplace(edges_.back().name, new_id);
  if (edge_id != nullptr) {
    *edge_id = new_id;
  }
  return Status::Ok();
}

Status RuntimeGraphBuilder::AddWeight(RuntimeWeightDesc weight) {
  if (finalized_) {
    return Fail(ErrorCode::kStateError,
                "runtime graph builder has already been finalized");
  }
  if (!IsValidGraphName(weight.ref)) {
    return Fail(ErrorCode::kInvalidArgument,
                "weight ref is empty or invalid: " + weight.ref);
  }
  if (!IsValidGraphName(weight.source_name)) {
    return Fail(
        ErrorCode::kInvalidArgument,
        "weight source_name is empty or invalid: " + weight.source_name);
  }
  if (weight_refs_.find(weight.ref) != weight_refs_.end()) {
    return Fail(ErrorCode::kInvalidArgument,
                "duplicate weight ref: " + weight.ref);
  }
  Status status = CheckTensorDesc(weight.tensor, "weight " + weight.ref);
  if (!status.ok()) {
    return Fail(status.code(), status.message());
  }
  weight_refs_.insert(weight.ref);
  weights_.push_back(std::move(weight));
  return Status::Ok();
}

Status RuntimeGraphBuilder::SetModelConfig(RuntimeModelConfig model_config) {
  if (finalized_) {
    return Fail(ErrorCode::kStateError,
                "runtime graph builder has already been finalized");
  }
  model_config_ = std::move(model_config);
  return Status::Ok();
}

Status RuntimeGraphBuilder::SetSchemaVersion(uint32_t major, uint32_t minor) {
  if (finalized_) {
    return Fail(ErrorCode::kStateError,
                "runtime graph builder has already been finalized");
  }
  if (major == 0) {
    return Fail(ErrorCode::kInvalidArgument,
                "schema major version must be positive");
  }
  schema_version_major_ = major;
  schema_version_minor_ = minor;
  return Status::Ok();
}

Status RuntimeGraphBuilder::SetFusionRulesVersion(std::string version) {
  if (finalized_) {
    return Fail(ErrorCode::kStateError,
                "runtime graph builder has already been finalized");
  }
  if (version.empty()) {
    return Fail(ErrorCode::kInvalidArgument,
                "fusion rules version must not be empty");
  }
  fusion_rules_version_ = std::move(version);
  return Status::Ok();
}

Status RuntimeGraphBuilder::SetInputNodes(std::vector<NodeId> node_ids) {
  if (finalized_) {
    return Fail(ErrorCode::kStateError,
                "runtime graph builder has already been finalized");
  }
  std::set<NodeId> unique_ids;
  for (NodeId node_id : node_ids) {
    if (node_id >= nodes_.size()) {
      return Fail(ErrorCode::kInvalidArgument, "input node id is out of range");
    }
    if (nodes_[node_id].op_type != OpType::kInput) {
      return Fail(ErrorCode::kInvalidArgument,
                  "input node id does not refer to an input node: " +
                      nodes_[node_id].name);
    }
    if (!unique_ids.insert(node_id).second) {
      return Fail(ErrorCode::kInvalidArgument,
                  "duplicate input node id: " + std::to_string(node_id));
    }
  }
  input_nodes_ = std::move(node_ids);
  return Status::Ok();
}

Status RuntimeGraphBuilder::SetOutputNodes(std::vector<NodeId> node_ids) {
  if (finalized_) {
    return Fail(ErrorCode::kStateError,
                "runtime graph builder has already been finalized");
  }
  std::set<NodeId> unique_ids;
  for (NodeId node_id : node_ids) {
    if (node_id >= nodes_.size()) {
      return Fail(ErrorCode::kInvalidArgument,
                  "output node id is out of range");
    }
    if (nodes_[node_id].op_type != OpType::kOutput) {
      return Fail(ErrorCode::kInvalidArgument,
                  "output node id does not refer to an output node: " +
                      nodes_[node_id].name);
    }
    if (!unique_ids.insert(node_id).second) {
      return Fail(ErrorCode::kInvalidArgument,
                  "duplicate output node id: " + std::to_string(node_id));
    }
  }
  output_nodes_ = std::move(node_ids);
  return Status::Ok();
}

Status RuntimeGraphBuilder::Finalize(std::unique_ptr<RuntimeGraph>* graph) {
  if (graph == nullptr) {
    return Fail(ErrorCode::kInvalidArgument,
                "runtime graph output pointer must not be null");
  }
  if (finalized_) {
    return Fail(ErrorCode::kStateError,
                "runtime graph builder has already been finalized");
  }
  if (input_nodes_.empty()) {
    return Fail(ErrorCode::kInvalidArgument,
                "runtime graph builder has no input nodes");
  }
  if (output_nodes_.empty()) {
    return Fail(ErrorCode::kInvalidArgument,
                "runtime graph builder has no output nodes");
  }

  std::vector<NodeId> producers(edges_.size(), kInvalidNodeId);
  std::vector<std::vector<NodeId>> consumers(edges_.size());
  for (NodeId node_id = 0; node_id < nodes_.size(); ++node_id) {
    const RuntimeNode& node = nodes_[node_id];
    std::set<EdgeId> local_inputs;
    std::set<EdgeId> local_outputs;
    for (EdgeId edge_id : node.input_edges) {
      if (edge_id >= edges_.size()) {
        return Fail(ErrorCode::kInvalidArgument,
                    "node references out-of-range input edge: " + node.name);
      }
      if (!local_inputs.insert(edge_id).second) {
        return Fail(ErrorCode::kInvalidArgument,
                    "node has duplicate input edge: " + node.name + " / " +
                        edges_[edge_id].name);
      }
      consumers[edge_id].push_back(node_id);
    }
    for (EdgeId edge_id : node.output_edges) {
      if (edge_id >= edges_.size()) {
        return Fail(ErrorCode::kInvalidArgument,
                    "node references out-of-range output edge: " + node.name);
      }
      if (!local_outputs.insert(edge_id).second) {
        return Fail(ErrorCode::kInvalidArgument,
                    "node has duplicate output edge: " + node.name + " / " +
                        edges_[edge_id].name);
      }
      if (edges_[edge_id].is_weight) {
        return Fail(ErrorCode::kInvalidArgument,
                    "weight edge cannot be produced by node: " + node.name +
                        " / " + edges_[edge_id].name);
      }
      if (producers[edge_id] != kInvalidNodeId) {
        return Fail(ErrorCode::kInvalidArgument,
                    "edge has multiple producers: " + edges_[edge_id].name);
      }
      producers[edge_id] = node_id;
    }
  }

  std::vector<RuntimeEdge> final_edges = edges_;
  for (size_t edge_id = 0; edge_id < final_edges.size(); ++edge_id) {
    RuntimeEdge& edge = final_edges[edge_id];
    if (edge.is_weight) {
      if (producers[edge_id] != kInvalidNodeId) {
        return Fail(ErrorCode::kInvalidArgument,
                    "weight edge has a producer: " + edge.name);
      }
    } else {
      if (producers[edge_id] == kInvalidNodeId) {
        return Fail(ErrorCode::kInvalidArgument,
                    "activation edge has no producer: " + edge.name);
      }
      edge.producer = producers[edge_id];
    }
    if (consumers[edge_id].empty()) {
      return Fail(ErrorCode::kInvalidArgument,
                  "edge has no consumers: " + edge.name);
    }
    edge.consumers = consumers[edge_id];
  }

  std::vector<size_t> indegree(nodes_.size(), 0);
  std::vector<std::vector<NodeId>> adjacency(nodes_.size());
  for (NodeId node_id = 0; node_id < nodes_.size(); ++node_id) {
    for (EdgeId edge_id : nodes_[node_id].input_edges) {
      if (edge_id < edges_.size() && !edges_[edge_id].is_weight) {
        ++indegree[node_id];
      }
    }
  }
  for (size_t edge_id = 0; edge_id < final_edges.size(); ++edge_id) {
    const RuntimeEdge& edge = final_edges[edge_id];
    if (edge.is_weight) {
      continue;
    }
    for (NodeId consumer_id : edge.consumers) {
      adjacency[edge.producer].push_back(consumer_id);
    }
  }
  std::priority_queue<NodeId, std::vector<NodeId>, std::greater<NodeId>> ready;
  for (NodeId node_id = 0; node_id < nodes_.size(); ++node_id) {
    if (indegree[node_id] == 0) {
      ready.push(node_id);
    }
  }
  std::vector<NodeId> topo_order;
  topo_order.reserve(nodes_.size());
  while (!ready.empty()) {
    const NodeId current = ready.top();
    ready.pop();
    topo_order.push_back(current);
    for (NodeId next : adjacency[current]) {
      if (--indegree[next] == 0) {
        ready.push(next);
      }
    }
  }
  if (topo_order.size() != nodes_.size()) {
    return Fail(ErrorCode::kInvalidArgument, "graph contains a cycle");
  }

  auto candidate = std::unique_ptr<RuntimeGraph>(new RuntimeGraph());
  candidate->schema_version_major_ = schema_version_major_;
  candidate->schema_version_minor_ = schema_version_minor_;
  candidate->fusion_rules_version_ = fusion_rules_version_;
  candidate->model_config_ = model_config_;
  candidate->nodes_ = nodes_;
  candidate->edges_ = std::move(final_edges);
  candidate->weights_ = weights_;
  candidate->topo_order_ = std::move(topo_order);
  candidate->input_nodes_ = input_nodes_;
  candidate->output_nodes_ = output_nodes_;
  candidate->node_kernels_.assign(candidate->nodes_.size(), nullptr);
  for (NodeId node_id = 0; node_id < candidate->nodes_.size(); ++node_id) {
    candidate->node_ids_.emplace(candidate->nodes_[node_id].name, node_id);
  }
  for (EdgeId edge_id = 0; edge_id < candidate->edges_.size(); ++edge_id) {
    candidate->edge_ids_.emplace(candidate->edges_[edge_id].name, edge_id);
  }
  for (size_t weight_index = 0; weight_index < candidate->weights_.size();
       ++weight_index) {
    candidate->weight_indices_.emplace(candidate->weights_[weight_index].ref,
                                       weight_index);
  }

  ValidationReport report;
  Status status = candidate->Validate(&report);
  validation_report_ = report;
  if (!status.ok()) {
    return Status::Error(status.code(), report.ToString());
  }

  candidate->finalized_ = true;
  finalized_ = true;
  *graph = std::move(candidate);
  return Status::Ok();
}

}  // namespace graph
}  // namespace mobilenet
