// 模块：RuntimeGraph 校验器实现
//
// 职责：
//   实现 RuntimeGraph -> ValidationGraph 视图转换、V1~V16 复用，以及
//   V17 Backend/Kernel/Arena/I/O 绑定校验。
//
// 边界：
//   - 不执行 Kernel、不读写设备内存内容；
//   - V17 不检查模型语义，只检查执行前置条件。

#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "graph/runtime_validator.h"
#include "graph/validation_common.h"

namespace mobilenet {
namespace graph {
namespace {

// 添加问题。
void AddIssue(ValidationReport* report, ErrorCode code, std::string message,
              std::string node_name = "", std::string edge_name = "",
              std::string binding_name = "") {
  report->AddIssue(ValidationIssue{code, std::move(message),
                                   std::move(node_name), std::move(edge_name),
                                   std::move(binding_name), ""});
}

// TensorDesc -> 校验视图。
ValidationTensor MakeTensorView(const TensorDesc& tensor) {
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

// RuntimeModelConfig -> 校验视图。
ValidationModelConfig MakeConfigView(const RuntimeModelConfig& config) {
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

// 构造校验视图。
ValidationGraph MakeValidationGraph(const RuntimeGraph& graph) {
  ValidationGraph view;
  view.schema_major = graph.schema_version_major();
  view.schema_minor = graph.schema_version_minor();
  view.fusion_rules_version = graph.fusion_rules_version();
  view.model_config = MakeConfigView(graph.model_config());
  view.nodes.reserve(graph.nodes().size());
  for (const RuntimeNode& node : graph.nodes()) {
    ValidationNode node_view;
    node_view.name = node.name;
    node_view.op_type = node.op_type;
    node_view.precision = DataTypeName(node.precision);
    node_view.kernel_key = node.kernel_key;
    node_view.fusion_rule_id = node.fusion_rule_id;
    node_view.binding_name = node.binding_name;
    node_view.copy_kind = node.copy_kind;
    node_view.fused_ops = node.fused_ops;
    node_view.attributes = node.attributes;
    node_view.has_host_tensor = node.host_tensor.has_value();
    if (node.host_tensor.has_value()) {
      node_view.host_tensor = MakeTensorView(*node.host_tensor);
    }
    for (EdgeId edge_id : node.input_edges) {
      if (edge_id < graph.edges().size()) {
        node_view.input_edges.push_back(graph.edges()[edge_id].name);
      }
    }
    for (EdgeId edge_id : node.output_edges) {
      if (edge_id < graph.edges().size()) {
        node_view.output_edges.push_back(graph.edges()[edge_id].name);
      }
    }
    view.nodes.push_back(std::move(node_view));
  }
  view.edges.reserve(graph.edges().size());
  for (const RuntimeEdge& edge : graph.edges()) {
    ValidationEdge edge_view;
    edge_view.name = edge.name;
    edge_view.tensor = MakeTensorView(edge.tensor);
    edge_view.producer = edge.producer < graph.nodes().size()
                             ? graph.nodes()[edge.producer].name
                             : "";
    for (NodeId consumer_id : edge.consumers) {
      if (consumer_id < graph.nodes().size()) {
        edge_view.consumers.push_back(graph.nodes()[consumer_id].name);
      }
    }
    edge_view.is_weight = edge.is_weight;
    edge_view.weight_ref = edge.weight_ref;
    view.edges.push_back(std::move(edge_view));
  }
  view.weights.reserve(graph.weights().size());
  for (const RuntimeWeightDesc& weight : graph.weights()) {
    ValidationWeight weight_view;
    weight_view.ref = weight.ref;
    weight_view.source_name = weight.source_name;
    weight_view.tensor = MakeTensorView(weight.tensor);
    weight_view.quantization = weight.tensor.quantization;
    weight_view.checksum = weight.checksum;
    view.weights.push_back(std::move(weight_view));
  }
  view.topo_order = graph.topo_order();
  view.input_nodes = graph.input_nodes();
  view.output_nodes = graph.output_nodes();
  return view;
}

// 检查 IoBinding 与 host TensorDesc 一致。
void CheckIoBinding(const IoBinding& binding, const TensorDesc& host_tensor,
                    std::string_view binding_name, ValidationReport* report) {
  if (binding.host_ptr == nullptr) {
    AddIssue(report, ErrorCode::kInvalidArgument,
             "I/O binding host_ptr is null", "", "", std::string(binding_name));
  }
  if (binding.bytes < host_tensor.size_bytes) {
    AddIssue(report, ErrorCode::kBufferError,
             "I/O binding is smaller than host tensor", "", "",
             std::string(binding_name));
  }
  if (binding.dtype != host_tensor.dtype) {
    AddIssue(report, ErrorCode::kUnsupportedDtype,
             "I/O binding dtype does not match host tensor", "", "",
             std::string(binding_name));
  }
  if (binding.shape.size() != host_tensor.shape.size()) {
    AddIssue(report, ErrorCode::kInvalidArgument,
             "I/O binding rank does not match host tensor", "", "",
             std::string(binding_name));
    return;
  }
  for (size_t index = 0; index < binding.shape.size(); ++index) {
    if (binding.shape[index].value != host_tensor.shape[index].value) {
      AddIssue(report, ErrorCode::kInvalidArgument,
               "I/O binding shape does not match host tensor", "", "",
               std::string(binding_name));
      return;
    }
  }
}

}  // namespace

Status RuntimeGraphValidator::Validate(const RuntimeGraph& graph,
                                       ValidationReport* report) {
  if (report == nullptr) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "validation report must not be null");
  }
  const ValidationGraph view = MakeValidationGraph(graph);
  return ValidateGraphView(view, RuleLookup{}, report);
}

Status RuntimeGraphValidator::ValidateExecution(const RuntimeGraph& graph,
                                                ForwardContext* context,
                                                ValidationReport* report) {
  if (report == nullptr) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "validation report must not be null");
  }
  report->Clear();
  if (!graph.finalized_) {
    AddIssue(report, ErrorCode::kStateError, "graph is not finalized");
  }
  if (graph.released_) {
    AddIssue(report, ErrorCode::kStateError, "graph has been released");
  }
  if (graph.backend_ == nullptr) {
    AddIssue(report, ErrorCode::kStateError, "backend is not bound");
  }
  if (graph.node_kernels_.size() != graph.nodes_.size()) {
    AddIssue(report, ErrorCode::kInternal,
             "kernel table size does not match node count");
  }
  for (NodeId node_id = 0; node_id < graph.nodes_.size(); ++node_id) {
    const RuntimeNode& node = graph.nodes_[node_id];
    if (IsComputeOp(node.op_type)) {
      if (node_id >= graph.node_kernels_.size() ||
          graph.node_kernels_[node_id] == nullptr) {
        AddIssue(report, ErrorCode::kStateError, "compute node has no kernel",
                 node.name);
      }
    }
  }
  for (const RuntimeEdge& edge : graph.edges_) {
    if (edge.arena_id < 0) {
      AddIssue(report, ErrorCode::kStateError, "edge has no arena binding", "",
               edge.name);
      continue;
    }
    const auto arena_iterator = graph.arenas_.find(edge.arena_id);
    if (arena_iterator == graph.arenas_.end()) {
      AddIssue(report, ErrorCode::kStateError, "edge arena is not bound", "",
               edge.name);
      continue;
    }
    const ArenaBinding& arena = arena_iterator->second;
    if (arena.base == nullptr || arena.bytes == 0) {
      AddIssue(report, ErrorCode::kBufferError, "edge arena is empty", "",
               edge.name);
      continue;
    }
    if (edge.memory_offset >
        std::numeric_limits<uint64_t>::max() - edge.tensor.size_bytes) {
      AddIssue(report, ErrorCode::kBufferError, "edge memory range overflows",
               "", edge.name);
      continue;
    }
    if (edge.memory_offset + edge.tensor.size_bytes > arena.bytes) {
      AddIssue(report, ErrorCode::kBufferError,
               "edge memory range exceeds arena", "", edge.name);
    }
  }

  if (context == nullptr) {
    AddIssue(report, ErrorCode::kInvalidArgument,
             "forward context must not be null");
  } else {
    if (context->backend != nullptr && context->backend != graph.backend_) {
      AddIssue(report, ErrorCode::kStateError,
               "forward context backend does not match graph backend");
    }
    for (NodeId node_id : graph.input_nodes_) {
      if (node_id >= graph.nodes_.size()) {
        continue;
      }
      const RuntimeNode& node = graph.nodes_[node_id];
      const auto binding_iterator =
          context->host_inputs.find(node.binding_name);
      if (binding_iterator == context->host_inputs.end()) {
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "input binding is missing", node.name, "", node.binding_name);
        continue;
      }
      if (!node.host_tensor.has_value()) {
        AddIssue(report, ErrorCode::kInternal, "input node has no host tensor",
                 node.name);
        continue;
      }
      CheckIoBinding(binding_iterator->second, *node.host_tensor,
                     node.binding_name, report);
    }
    for (NodeId node_id : graph.output_nodes_) {
      if (node_id >= graph.nodes_.size()) {
        continue;
      }
      const RuntimeNode& node = graph.nodes_[node_id];
      const auto binding_iterator =
          context->host_outputs.find(node.binding_name);
      if (binding_iterator == context->host_outputs.end()) {
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "output binding is missing", node.name, "", node.binding_name);
        continue;
      }
      if (!node.host_tensor.has_value()) {
        AddIssue(report, ErrorCode::kInternal, "output node has no host tensor",
                 node.name);
        continue;
      }
      CheckIoBinding(binding_iterator->second, *node.host_tensor,
                     node.binding_name, report);
    }
  }

  if (report->ok()) {
    return Status::Ok();
  }
  return Status::Error(report->issues().front().code,
                       "execution validation failed");
}

}  // namespace graph
}  // namespace mobilenet
