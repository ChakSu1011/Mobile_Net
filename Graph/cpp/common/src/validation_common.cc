// 模块：Graph 公共校验规则实现
//
// 职责：
//   实现 V1~V16 的公共校验逻辑，供 Graph IR 与 RuntimeGraph 共用。
//
// 边界：
//   - 不执行任何张量数据访问；
//   - 不修改输入视图；
//   - V17 运行时校验不在本文件。

#include "graph/validation_common.h"

#include <limits>
#include <sstream>
#include <unordered_map>

namespace mobilenet {
namespace graph {
namespace {

// 添加一条校验问题。
void AddIssue(ValidationReport* report, ErrorCode code, std::string message,
              std::string node_name = "", std::string edge_name = "",
              std::string binding_name = "", std::string rule_id = "") {
  report->AddIssue(ValidationIssue{
      code, std::move(message), std::move(node_name), std::move(edge_name),
      std::move(binding_name), std::move(rule_id)});
}

// 判断字符串是否非空且为合法图名称。
bool ValidName(std::string_view name) { return IsValidGraphName(name); }

// 判断属性值中的浮点数是否有限。
bool AttributeIsFinite(const AttributeValue& value) {
  if (const auto* number = std::get_if<double>(&value)) {
    return std::isfinite(*number);
  }
  if (const auto* numbers = std::get_if<std::vector<double>>(&value)) {
    for (double number : *numbers) {
      if (!std::isfinite(number)) {
        return false;
      }
    }
  }
  return true;
}

// 计算形状元素个数，溢出返回 false。
bool ElementCount(const std::vector<Dim>& shape, uint64_t* count) {
  uint64_t result = 1;
  for (const Dim& dim : shape) {
    if (dim.value < 0) {
      return false;
    }
    const uint64_t value = static_cast<uint64_t>(dim.value);
    if (value != 0 && result > std::numeric_limits<uint64_t>::max() / value) {
      return false;
    }
    result *= value;
  }
  *count = result;
  return true;
}

// 判断两个形状的维度值与符号是否完全一致。
bool SameShape(const std::vector<Dim>& left, const std::vector<Dim>& right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (size_t index = 0; index < left.size(); ++index) {
    if (left[index].is_symbolic != right[index].is_symbolic ||
        left[index].value != right[index].value ||
        left[index].symbol != right[index].symbol) {
      return false;
    }
  }
  return true;
}

// 判断向量中是否包含指定字符串。
bool Contains(const std::vector<std::string>& values, std::string_view target) {
  return std::find(values.begin(), values.end(), target) != values.end();
}

// 校验单个张量元数据。
//
// 参数：
//   tensor：待检查张量；
//   is_weight：是否为权重（权重不允许符号维）；
//   node_name/edge_name：诊断上下文；
//   report：输出报告。
void CheckTensor(const ValidationTensor& tensor, bool is_weight,
                 std::string_view node_name, std::string_view edge_name,
                 ValidationReport* report) {
  if (!tensor.defined) {
    AddIssue(report, ErrorCode::kInvalidArgument, "tensor is undefined",
             std::string(node_name), std::string(edge_name));
    return;
  }
  const DataType dtype = DataTypeFromName(tensor.dtype);
  if (dtype == DataType::kUnknown) {
    AddIssue(report, ErrorCode::kUnsupportedDtype,
             "tensor dtype is unknown: " + tensor.dtype, std::string(node_name),
             std::string(edge_name));
  }
  if (tensor.shape.size() != tensor.stride.size()) {
    AddIssue(report, ErrorCode::kInvalidArgument,
             "tensor shape rank does not match stride rank",
             std::string(node_name), std::string(edge_name));
  }
  std::map<std::string, int64_t> symbol_values;
  for (const Dim& dim : tensor.shape) {
    if (dim.value < 0) {
      AddIssue(report, ErrorCode::kInvalidArgument,
               "tensor dimension is negative", std::string(node_name),
               std::string(edge_name));
    }
    if (dim.is_symbolic) {
      if (dim.symbol.empty()) {
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "symbolic dimension has empty symbol", std::string(node_name),
                 std::string(edge_name));
      } else {
        auto iterator = symbol_values.find(dim.symbol);
        if (iterator != symbol_values.end() && iterator->second != dim.value) {
          AddIssue(report, ErrorCode::kInvalidArgument,
                   "symbol " + dim.symbol + " has inconsistent example values",
                   std::string(node_name), std::string(edge_name));
        } else {
          symbol_values.emplace(dim.symbol, dim.value);
        }
      }
      if (is_weight) {
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "weight tensor contains symbolic dimension",
                 std::string(node_name), std::string(edge_name));
      }
    } else if (!dim.symbol.empty()) {
      AddIssue(report, ErrorCode::kInvalidArgument,
               "constant dimension has non-empty symbol",
               std::string(node_name), std::string(edge_name));
    }
  }
  for (int64_t stride_value : tensor.stride) {
    if (stride_value < 0) {
      AddIssue(report, ErrorCode::kInvalidArgument, "tensor stride is negative",
               std::string(node_name), std::string(edge_name));
      break;
    }
  }
  if (tensor.layout != Layout::kRowMajor &&
      tensor.layout != Layout::kColumnMajor) {
    AddIssue(report, ErrorCode::kUnsupportedLayout, "tensor layout is unknown",
             std::string(node_name), std::string(edge_name));
  }
  if (tensor.layout == Layout::kColumnMajor && tensor.shape.size() != 2) {
    AddIssue(report, ErrorCode::kUnsupportedLayout,
             "column-major tensor must be 2-D", std::string(node_name),
             std::string(edge_name));
  }
  if (tensor.device != DeviceType::kHost &&
      tensor.device != DeviceType::kDevice) {
    AddIssue(report, ErrorCode::kInvalidArgument, "tensor device is unknown",
             std::string(node_name), std::string(edge_name));
  }
  if (!tensor.quantization.has_value()) {
    uint64_t elements = 0;
    if (!ElementCount(tensor.shape, &elements)) {
      AddIssue(report, ErrorCode::kInvalidArgument,
               "tensor element count overflows", std::string(node_name),
               std::string(edge_name));
    } else if (dtype != DataType::kUnknown) {
      const uint64_t element_size = DataTypeSize(dtype);
      if (element_size != 0 && tensor.size_bytes != elements * element_size) {
        AddIssue(report, ErrorCode::kBufferError,
                 "tensor size_bytes does not match shape and dtype",
                 std::string(node_name), std::string(edge_name));
      }
    }
  } else {
    const QuantizationDesc& quantization = *tensor.quantization;
    if (quantization.storage_dtype == DataType::kUnknown) {
      AddIssue(report, ErrorCode::kUnsupportedDtype,
               "quantization storage_dtype is unknown", std::string(node_name),
               std::string(edge_name));
    }
    if (quantization.granularity == QuantGranularity::kNone) {
      AddIssue(report, ErrorCode::kInvalidArgument,
               "quantization granularity is none", std::string(node_name),
               std::string(edge_name));
    }
    if (quantization.granularity == QuantGranularity::kPerGroup &&
        quantization.group_size <= 0) {
      AddIssue(report, ErrorCode::kInvalidArgument,
               "per-group quantization requires positive group_size",
               std::string(node_name), std::string(edge_name));
    }
    if (quantization.scales.empty()) {
      AddIssue(report, ErrorCode::kInvalidArgument,
               "quantization scales are empty", std::string(node_name),
               std::string(edge_name));
    }
    for (double scale : quantization.scales) {
      if (!std::isfinite(scale)) {
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "quantization scale is not finite", std::string(node_name),
                 std::string(edge_name));
        break;
      }
    }
  }
}

// 判断节点是否为 I/O 边界节点。
bool IsIoNode(OpType op_type) {
  return op_type == OpType::kInput || op_type == OpType::kOutput;
}

// 判断两个张量元数据是否描述同一权重载荷。
bool SameWeightTensor(const ValidationTensor& left,
                      const ValidationTensor& right) {
  return left.dtype == right.dtype && SameShape(left.shape, right.shape) &&
         left.stride == right.stride && left.size_bytes == right.size_bytes &&
         left.offset == right.offset;
}

}  // namespace

DataType DataTypeFromName(std::string_view name) {
  if (name == "float32") {
    return DataType::kFloat32;
  }
  if (name == "float16") {
    return DataType::kFloat16;
  }
  if (name == "bfloat16") {
    return DataType::kBFloat16;
  }
  if (name == "int8") {
    return DataType::kInt8;
  }
  if (name == "uint8") {
    return DataType::kUInt8;
  }
  if (name == "int32") {
    return DataType::kInt32;
  }
  if (name == "int64") {
    return DataType::kInt64;
  }
  if (name == "bool") {
    return DataType::kBool;
  }
  if (name == "float64") {
    return DataType::kFloat64;
  }
  return DataType::kUnknown;
}

Status ValidateGraphView(const ValidationGraph& graph,
                         const RuleLookup& rule_lookup,
                         ValidationReport* report) {
  if (report == nullptr) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "validation report must not be null");
  }
  report->Clear();

  // V15：schema 与规则集版本。
  if (graph.schema_major != 1u || graph.schema_minor > 0u) {
    AddIssue(report, ErrorCode::kVersionMismatch,
             "unsupported graph schema version");
  }
  if (graph.fusion_rules_version.empty()) {
    AddIssue(report, ErrorCode::kVersionMismatch,
             "fusion_rules_version is empty");
  }

  // V16：模型配置。
  const ValidationModelConfig& config = graph.model_config;
  if (!ValidName(config.model_family)) {
    AddIssue(report, ErrorCode::kInvalidArgument,
             "model_family is empty or invalid");
  }
  if (config.num_layers <= 0) {
    AddIssue(report, ErrorCode::kInvalidArgument,
             "num_layers must be positive");
  }
  if (config.hidden_size <= 0) {
    AddIssue(report, ErrorCode::kInvalidArgument,
             "hidden_size must be positive");
  }
  if (config.intermediate_size <= 0) {
    AddIssue(report, ErrorCode::kInvalidArgument,
             "intermediate_size must be positive");
  }
  if (config.num_attention_heads <= 0) {
    AddIssue(report, ErrorCode::kInvalidArgument,
             "num_attention_heads must be positive");
  }
  if (config.num_kv_heads <= 0) {
    AddIssue(report, ErrorCode::kInvalidArgument,
             "num_kv_heads must be positive");
  }
  if (config.num_attention_heads > 0 && config.num_kv_heads > 0 &&
      config.num_attention_heads % config.num_kv_heads != 0) {
    AddIssue(report, ErrorCode::kInvalidArgument,
             "num_attention_heads must be divisible by num_kv_heads");
  }
  if (config.num_kv_heads > config.num_attention_heads) {
    AddIssue(report, ErrorCode::kInvalidArgument,
             "num_kv_heads must not exceed num_attention_heads");
  }
  if (config.head_dim <= 0) {
    AddIssue(report, ErrorCode::kInvalidArgument, "head_dim must be positive");
  }
  if (config.vocab_size <= 0) {
    AddIssue(report, ErrorCode::kInvalidArgument,
             "vocab_size must be positive");
  }
  if (config.max_context_length <= 0) {
    AddIssue(report, ErrorCode::kInvalidArgument,
             "max_context_length must be positive");
  }
  if (!(config.norm_eps > 0.0) || !std::isfinite(config.norm_eps)) {
    AddIssue(report, ErrorCode::kInvalidArgument,
             "norm_eps must be positive and finite");
  }
  if (config.activation.empty()) {
    AddIssue(report, ErrorCode::kInvalidArgument, "activation is empty");
  }
  if (config.dtype.empty() ||
      DataTypeFromName(config.dtype) == DataType::kUnknown) {
    AddIssue(report, ErrorCode::kUnsupportedDtype,
             "model dtype is empty or unknown");
  }
  if (!(config.rope.theta > 0.0) || !std::isfinite(config.rope.theta)) {
    AddIssue(report, ErrorCode::kInvalidArgument,
             "rope theta must be positive and finite");
  }
  if (!(config.rope.scaling_factor > 0.0) ||
      !std::isfinite(config.rope.scaling_factor)) {
    AddIssue(report, ErrorCode::kInvalidArgument,
             "rope scaling_factor must be positive and finite");
  }

  // V4：输入/输出集合非空。
  if (graph.nodes.empty()) {
    AddIssue(report, ErrorCode::kInvalidArgument, "graph has no nodes");
  }
  if (graph.input_nodes.empty()) {
    AddIssue(report, ErrorCode::kInvalidArgument, "graph has no input nodes");
  }
  if (graph.output_nodes.empty()) {
    AddIssue(report, ErrorCode::kInvalidArgument, "graph has no output nodes");
  }

  // V1/V2：节点与边名称、唯一性。
  std::map<std::string, size_t> node_index_by_name;
  std::map<std::string, size_t> edge_index_by_name;
  std::map<std::string, size_t> weight_index_by_ref;
  std::set<std::string> binding_names;
  for (size_t index = 0; index < graph.nodes.size(); ++index) {
    const ValidationNode& node = graph.nodes[index];
    if (!ValidName(node.name)) {
      AddIssue(report, ErrorCode::kInvalidArgument, "node name is invalid",
               node.name);
    }
    if (!node_index_by_name.emplace(node.name, index).second) {
      AddIssue(report, ErrorCode::kInvalidArgument, "duplicate node name",
               node.name);
    }
    if (node.op_type == OpType::kUnknown) {
      AddIssue(report, ErrorCode::kInvalidArgument, "node op_type is unknown",
               node.name);
    }
    if (!node.binding_name.empty()) {
      if (!ValidName(node.binding_name)) {
        AddIssue(report, ErrorCode::kInvalidArgument, "binding_name is invalid",
                 node.name, "", node.binding_name);
      }
      if (!binding_names.insert(node.binding_name).second) {
        AddIssue(report, ErrorCode::kInvalidArgument, "duplicate binding_name",
                 node.name, "", node.binding_name);
      }
    }
    if (!node.fusion_rule_id.empty() && !ValidName(node.fusion_rule_id)) {
      AddIssue(report, ErrorCode::kInvalidArgument, "fusion_rule_id is invalid",
               node.name, "", node.binding_name, node.fusion_rule_id);
    }
    if (!node.kernel_key.empty() && !ValidName(node.kernel_key)) {
      AddIssue(report, ErrorCode::kInvalidArgument, "kernel_key is invalid",
               node.name);
    }
  }

  for (size_t index = 0; index < graph.edges.size(); ++index) {
    const ValidationEdge& edge = graph.edges[index];
    if (!ValidName(edge.name)) {
      AddIssue(report, ErrorCode::kInvalidArgument, "edge name is invalid", "",
               edge.name);
    }
    if (!edge_index_by_name.emplace(edge.name, index).second) {
      AddIssue(report, ErrorCode::kInvalidArgument, "duplicate edge name", "",
               edge.name);
    }
  }
  for (size_t index = 0; index < graph.weights.size(); ++index) {
    const ValidationWeight& weight = graph.weights[index];
    if (!ValidName(weight.ref)) {
      AddIssue(report, ErrorCode::kInvalidArgument, "weight ref is invalid");
    }
    if (!weight_index_by_ref.emplace(weight.ref, index).second) {
      AddIssue(report, ErrorCode::kInvalidArgument, "duplicate weight ref");
    }
    if (weight.source_name.empty()) {
      AddIssue(report, ErrorCode::kInvalidArgument,
               "weight source_name is empty");
    }
    CheckTensor(weight.tensor, true, "", "", report);
    if (weight.quantization.has_value() &&
        weight.quantization->storage_dtype == DataType::kUnknown) {
      AddIssue(report, ErrorCode::kUnsupportedDtype,
               "weight quantization storage_dtype is unknown");
    }
  }

  // V7/V8/V10：边引用、权重边、张量元数据。
  for (const ValidationNode& node : graph.nodes) {
    std::set<std::string> input_names;
    std::set<std::string> output_names;
    for (const std::string& edge_name : node.input_edges) {
      if (!input_names.insert(edge_name).second) {
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "duplicate input edge on node", node.name, edge_name);
      }
      if (edge_index_by_name.find(edge_name) == edge_index_by_name.end()) {
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "node references missing input edge", node.name, edge_name);
      }
    }
    for (const std::string& edge_name : node.output_edges) {
      if (!output_names.insert(edge_name).second) {
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "duplicate output edge on node", node.name, edge_name);
      }
      if (edge_index_by_name.find(edge_name) == edge_index_by_name.end()) {
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "node references missing output edge", node.name, edge_name);
      }
    }
    for (const FusedOp& fused_op : node.fused_ops) {
      if (fused_op.source_name.empty() || fused_op.op_kind.empty()) {
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "fused op has empty source_name or op_kind", node.name);
      }
      for (const auto& attribute : fused_op.attributes) {
        if (attribute.first.empty() || !AttributeIsFinite(attribute.second)) {
          AddIssue(report, ErrorCode::kInvalidArgument,
                   "fused op attribute is invalid", node.name);
        }
      }
    }
    for (const auto& attribute : node.attributes) {
      if (attribute.first.empty() || !AttributeIsFinite(attribute.second)) {
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "node attribute is invalid", node.name);
      }
    }
  }

  for (const ValidationEdge& edge : graph.edges) {
    CheckTensor(edge.tensor, edge.is_weight, "", edge.name, report);
    if (edge.is_weight) {
      if (!edge.producer.empty()) {
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "weight edge must not have producer", "", edge.name);
      }
      if (edge.weight_ref.empty()) {
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "weight edge has empty weight_ref", "", edge.name);
      } else {
        auto weight_iterator = weight_index_by_ref.find(edge.weight_ref);
        if (weight_iterator == weight_index_by_ref.end()) {
          AddIssue(report, ErrorCode::kInvalidArgument,
                   "weight edge references missing weight", "", edge.name);
        } else {
          const ValidationWeight& weight =
              graph.weights[weight_iterator->second];
          if (!SameWeightTensor(edge.tensor, weight.tensor)) {
            AddIssue(report, ErrorCode::kBufferError,
                     "weight edge tensor does not match weight descriptor", "",
                     edge.name);
          }
          if (edge.tensor.storage_key != nullptr &&
              weight.tensor.storage_key != nullptr &&
              edge.tensor.storage_key != weight.tensor.storage_key) {
            AddIssue(report, ErrorCode::kInvalidArgument,
                     "weight edge does not share weight storage", "",
                     edge.name);
          }
        }
      }
      if (edge.consumers.empty()) {
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "weight edge has no consumer", "", edge.name);
      }
    } else {
      if (edge.producer.empty()) {
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "activation edge has no producer", "", edge.name);
      } else {
        auto producer_iterator = node_index_by_name.find(edge.producer);
        if (producer_iterator == node_index_by_name.end()) {
          AddIssue(report, ErrorCode::kInvalidArgument,
                   "edge producer does not exist", "", edge.name);
        } else if (!Contains(
                       graph.nodes[producer_iterator->second].output_edges,
                       edge.name)) {
          AddIssue(report, ErrorCode::kInvalidArgument,
                   "producer does not list edge as output", "", edge.name);
        }
      }
      if (edge.weight_ref.empty() == false) {
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "activation edge must not have weight_ref", "", edge.name);
      }
      if (edge.consumers.empty()) {
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "activation edge has no consumer", "", edge.name);
      }
      for (const std::string& consumer_name : edge.consumers) {
        auto consumer_iterator = node_index_by_name.find(consumer_name);
        if (consumer_iterator == node_index_by_name.end()) {
          AddIssue(report, ErrorCode::kInvalidArgument,
                   "edge consumer does not exist", "", edge.name);
        } else if (!Contains(graph.nodes[consumer_iterator->second].input_edges,
                             edge.name)) {
          AddIssue(report, ErrorCode::kInvalidArgument,
                   "consumer does not list edge as input", "", edge.name);
        }
      }
      if (!edge.producer.empty()) {
        for (const std::string& consumer_name : edge.consumers) {
          if (consumer_name == edge.producer) {
            AddIssue(report, ErrorCode::kInvalidArgument,
                     "edge producer is also its consumer", "", edge.name);
          }
        }
      }
    }
  }

  // V5/V6/V11/V12/V13：节点语义。
  size_t sampling_count = 0;
  size_t lm_head_count = 0;
  size_t sampling_index = 0;
  size_t lm_head_index = 0;
  for (size_t index = 0; index < graph.nodes.size(); ++index) {
    const ValidationNode& node = graph.nodes[index];
    if (node.op_type == OpType::kSampling) {
      ++sampling_count;
      sampling_index = index;
    } else if (node.op_type == OpType::kLmHead) {
      ++lm_head_count;
      lm_head_index = index;
    }
    if (IsIoNode(node.op_type)) {
      if (!node.fused_ops.empty()) {
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "I/O node must not have fused_ops", node.name);
      }
      if (!node.kernel_key.empty()) {
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "I/O node must not have kernel_key", node.name);
      }
      if (!node.has_host_tensor) {
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "I/O node must have host_tensor", node.name);
      } else {
        CheckTensor(node.host_tensor, false, node.name, "", report);
        if (node.host_tensor.device != DeviceType::kHost) {
          AddIssue(report, ErrorCode::kInvalidArgument,
                   "I/O host_tensor must be on host", node.name);
        }
      }
      if (node.precision.empty() ||
          DataTypeFromName(node.precision) == DataType::kUnknown) {
        AddIssue(report, ErrorCode::kUnsupportedDtype,
                 "I/O node precision is unknown", node.name);
      }
      const std::vector<std::string>& io_edges =
          node.op_type == OpType::kInput ? node.output_edges : node.input_edges;
      if (io_edges.empty()) {
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "I/O node has no connected edge", node.name);
      }
      for (const std::string& edge_name : io_edges) {
        auto edge_iterator = edge_index_by_name.find(edge_name);
        if (edge_iterator == edge_index_by_name.end()) {
          continue;
        }
        const ValidationEdge& edge = graph.edges[edge_iterator->second];
        if (edge.is_weight) {
          AddIssue(report, ErrorCode::kInvalidArgument,
                   "I/O node must not connect a weight edge", node.name,
                   edge.name);
        }
        if (node.has_host_tensor) {
          if (edge.tensor.dtype != node.host_tensor.dtype ||
              !SameShape(edge.tensor.shape, node.host_tensor.shape) ||
              edge.tensor.size_bytes != node.host_tensor.size_bytes) {
            AddIssue(report, ErrorCode::kInvalidArgument,
                     "I/O edge tensor does not match host_tensor", node.name,
                     edge.name);
          }
        }
        if (edge.tensor.device != DeviceType::kDevice) {
          AddIssue(report, ErrorCode::kInvalidArgument,
                   "I/O edge tensor must live on device", node.name, edge.name);
        }
      }
      if (node.op_type == OpType::kInput) {
        if (node.copy_kind != CopyKind::kH2D) {
          AddIssue(report, ErrorCode::kInvalidArgument,
                   "input node copy_kind must be kH2D", node.name);
        }
        if (!node.input_edges.empty()) {
          AddIssue(report, ErrorCode::kInvalidArgument,
                   "input node must not have input edges", node.name);
        }
      } else {
        if (node.copy_kind != CopyKind::kD2H) {
          AddIssue(report, ErrorCode::kInvalidArgument,
                   "output node copy_kind must be kD2H", node.name);
        }
        if (!node.output_edges.empty()) {
          AddIssue(report, ErrorCode::kInvalidArgument,
                   "output node must not have output edges", node.name);
        }
        if (node.input_edges.size() != 1) {
          AddIssue(report, ErrorCode::kInvalidArgument,
                   "output node must have exactly one input edge", node.name);
        } else {
          auto edge_iterator =
              edge_index_by_name.find(node.input_edges.front());
          if (edge_iterator != edge_index_by_name.end()) {
            const ValidationEdge& edge = graph.edges[edge_iterator->second];
            if (!edge.producer.empty() &&
                node_index_by_name.find(edge.producer) !=
                    node_index_by_name.end()) {
              const OpType producer_op =
                  graph.nodes[node_index_by_name[edge.producer]].op_type;
              if (!IsComputeOp(producer_op)) {
                AddIssue(report, ErrorCode::kInvalidArgument,
                         "output node input must be produced by compute node",
                         node.name, edge.name);
              }
            }
          }
        }
      }
      continue;
    }

    // 计算节点。
    if (!node.binding_name.empty() || node.copy_kind != CopyKind::kNone) {
      AddIssue(report, ErrorCode::kInvalidArgument,
               "compute node must not have binding/copy_kind", node.name);
    }
    if (node.input_edges.empty()) {
      AddIssue(report, ErrorCode::kInvalidArgument,
               "compute node has no input edges", node.name);
    }
    if (node.output_edges.empty()) {
      AddIssue(report, ErrorCode::kInvalidArgument,
               "compute node has no output edges", node.name);
    }
    if (node.fused_ops.empty()) {
      AddIssue(report, ErrorCode::kInvalidArgument,
               "compute node has no fused_ops", node.name);
    }
    if (node.kernel_key.empty()) {
      AddIssue(report, ErrorCode::kInvalidArgument,
               "compute node has no kernel_key", node.name);
    }
    if (node.precision.empty() ||
        DataTypeFromName(node.precision) == DataType::kUnknown) {
      AddIssue(report, ErrorCode::kUnsupportedDtype,
               "compute node precision is unknown", node.name);
    }
    for (const std::string& edge_name : node.output_edges) {
      auto edge_iterator = edge_index_by_name.find(edge_name);
      if (edge_iterator == edge_index_by_name.end()) {
        continue;
      }
      const ValidationEdge& edge = graph.edges[edge_iterator->second];
      if (edge.is_weight) {
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "compute node output must not be weight edge", node.name,
                 edge.name);
      }
      if (!node.precision.empty() && edge.tensor.dtype != node.precision) {
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "node precision does not match output edge dtype", node.name,
                 edge.name);
      }
    }

    // V13：融合规则一致性。
    if (!node.fusion_rule_id.empty()) {
      if (rule_lookup) {
        const RuleView* rule = rule_lookup(node.fusion_rule_id);
        if (rule == nullptr) {
          AddIssue(report, ErrorCode::kInvalidArgument, "fusion rule not found",
                   node.name, "", "", node.fusion_rule_id);
        } else {
          if (rule->model_family != config.model_family) {
            AddIssue(report, ErrorCode::kInvalidArgument,
                     "fusion rule model_family mismatch", node.name, "", "",
                     node.fusion_rule_id);
          }
          if (rule->op_type != node.op_type) {
            AddIssue(report, ErrorCode::kInvalidArgument,
                     "fusion rule op_type mismatch", node.name, "", "",
                     node.fusion_rule_id);
          }
          std::vector<std::string> actual_op_kinds;
          actual_op_kinds.reserve(node.fused_ops.size());
          for (const FusedOp& fused_op : node.fused_ops) {
            actual_op_kinds.push_back(fused_op.op_kind);
          }
          if (actual_op_kinds != rule->required_op_kinds) {
            AddIssue(report, ErrorCode::kInvalidArgument,
                     "fusion rule op sequence mismatch", node.name, "", "",
                     node.fusion_rule_id);
          }
          if (!rule->kernel_key.empty() &&
              rule->kernel_key != node.kernel_key) {
            AddIssue(report, ErrorCode::kInvalidArgument,
                     "fusion rule kernel_key mismatch", node.name, "", "",
                     node.fusion_rule_id);
          }
          for (const auto& required : rule->required_attributes) {
            const auto attribute_iterator =
                node.attributes.find(required.first);
            if (attribute_iterator == node.attributes.end() ||
                attribute_iterator->second != required.second) {
              AddIssue(
                  report, ErrorCode::kInvalidArgument,
                  "fusion rule required attribute missing or mismatched: " +
                      required.first,
                  node.name, "", "", node.fusion_rule_id);
            }
          }
        }
      }
    }
  }

  // V12：LM Head 与 Sampling 约束。
  if (sampling_count > 1) {
    AddIssue(report, ErrorCode::kInvalidArgument,
             "graph contains more than one sampling node");
  }
  if ((lm_head_count > 0 || sampling_count > 0) && sampling_count != 1) {
    AddIssue(report, ErrorCode::kInvalidArgument,
             "graph with lm_head/sampling must contain exactly one sampling "
             "node");
  }
  if (lm_head_count > 0 && sampling_count == 1 &&
      lm_head_index > sampling_index) {
    AddIssue(report, ErrorCode::kInvalidArgument,
             "lm_head must appear before sampling in node order");
  }

  // V9：无环性（独立 Kahn 检测）。
  {
    std::vector<size_t> indegree(graph.nodes.size(), 0);
    std::vector<std::vector<size_t>> adjacency(graph.nodes.size());
    for (const ValidationEdge& edge : graph.edges) {
      if (edge.is_weight || edge.producer.empty()) {
        continue;
      }
      auto producer_iterator = node_index_by_name.find(edge.producer);
      if (producer_iterator == node_index_by_name.end()) {
        continue;
      }
      for (const std::string& consumer_name : edge.consumers) {
        auto consumer_iterator = node_index_by_name.find(consumer_name);
        if (consumer_iterator == node_index_by_name.end()) {
          continue;
        }
        adjacency[producer_iterator->second].push_back(
            consumer_iterator->second);
        ++indegree[consumer_iterator->second];
      }
    }
    std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>>
        ready;
    for (size_t index = 0; index < indegree.size(); ++index) {
      if (indegree[index] == 0) {
        ready.push(index);
      }
    }
    size_t processed = 0;
    while (!ready.empty()) {
      const size_t current = ready.top();
      ready.pop();
      ++processed;
      for (size_t next : adjacency[current]) {
        if (--indegree[next] == 0) {
          ready.push(next);
        }
      }
    }
    if (processed != graph.nodes.size()) {
      AddIssue(report, ErrorCode::kInvalidArgument, "graph contains a cycle");
    }
  }

  // V14：拓扑序。
  if (graph.topo_order.size() != graph.nodes.size()) {
    AddIssue(report, ErrorCode::kInvalidArgument,
             "topo_order size does not match node count");
  } else {
    std::vector<size_t> topo_index(graph.nodes.size(), graph.nodes.size());
    bool valid = true;
    for (size_t position = 0; position < graph.topo_order.size(); ++position) {
      const NodeId node_id = graph.topo_order[position];
      if (node_id >= graph.nodes.size() ||
          topo_index[node_id] != graph.nodes.size()) {
        valid = false;
        AddIssue(report, ErrorCode::kInvalidArgument,
                 "topo_order contains invalid or duplicate node id");
        continue;
      }
      topo_index[node_id] = position;
    }
    if (valid) {
      for (const ValidationEdge& edge : graph.edges) {
        if (edge.is_weight || edge.producer.empty()) {
          continue;
        }
        auto producer_iterator = node_index_by_name.find(edge.producer);
        if (producer_iterator == node_index_by_name.end()) {
          continue;
        }
        const size_t producer_position = topo_index[producer_iterator->second];
        for (const std::string& consumer_name : edge.consumers) {
          auto consumer_iterator = node_index_by_name.find(consumer_name);
          if (consumer_iterator == node_index_by_name.end()) {
            continue;
          }
          if (producer_position >= topo_index[consumer_iterator->second]) {
            AddIssue(report, ErrorCode::kInvalidArgument,
                     "topo_order violates edge direction", "", edge.name);
          }
        }
      }
      size_t first_compute = graph.nodes.size();
      size_t last_compute = 0;
      for (size_t index = 0; index < graph.nodes.size(); ++index) {
        if (IsComputeOp(graph.nodes[index].op_type)) {
          first_compute = std::min(first_compute, topo_index[index]);
          last_compute = std::max(last_compute, topo_index[index]);
        }
      }
      if (first_compute != graph.nodes.size()) {
        for (NodeId input_id : graph.input_nodes) {
          if (input_id < graph.nodes.size() &&
              topo_index[input_id] >= first_compute) {
            AddIssue(report, ErrorCode::kInvalidArgument,
                     "input node must precede all compute nodes",
                     graph.nodes[input_id].name);
          }
        }
        for (NodeId output_id : graph.output_nodes) {
          if (output_id < graph.nodes.size() &&
              topo_index[output_id] <= last_compute) {
            AddIssue(report, ErrorCode::kInvalidArgument,
                     "output node must follow all compute nodes",
                     graph.nodes[output_id].name);
          }
        }
      }
    }
  }
  for (NodeId input_id : graph.input_nodes) {
    if (input_id >= graph.nodes.size()) {
      AddIssue(report, ErrorCode::kInvalidArgument,
               "input_nodes contains invalid node id");
    } else if (graph.nodes[input_id].op_type != OpType::kInput) {
      AddIssue(report, ErrorCode::kInvalidArgument,
               "input_nodes contains non-input node",
               graph.nodes[input_id].name);
    }
  }
  for (NodeId output_id : graph.output_nodes) {
    if (output_id >= graph.nodes.size()) {
      AddIssue(report, ErrorCode::kInvalidArgument,
               "output_nodes contains invalid node id");
    } else if (graph.nodes[output_id].op_type != OpType::kOutput) {
      AddIssue(report, ErrorCode::kInvalidArgument,
               "output_nodes contains non-output node",
               graph.nodes[output_id].name);
    }
  }

  if (report->ok()) {
    return Status::Ok();
  }
  return Status::Error(report->issues().front().code,
                       "graph validation failed");
}

}  // namespace graph
}  // namespace mobilenet
