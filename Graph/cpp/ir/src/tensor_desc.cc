// 模块：Graph IR 到 TensorDesc 转换实现
//
// 职责：
//   实现 dtype 映射、张量元数据视图、ToTensorDesc、Graph IR -> RuntimeGraph
//   转换与（LibTorch 构建下的）序列化前张量规范化。
//
// 边界：
//   - 只读取 at::Tensor 元数据，不读取元素值；
//   - 不分配 arena、不绑定 Kernel；
//   - 不修改源 Graph。

#include "graph/tensor_desc.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "graph/graph.h"
#include "graph/runtime_graph.h"
#include "graph/runtime_graph_builder.h"

namespace mobilenet {
namespace graph {
namespace {

// 从 at::Tensor 推导 Runtime 布局。
//
// 规则：
//   - 2-D 且 stride[0] == 1、stride[1] == size(0) 视为列主序；
//   - 其余视为行主序（允许自定义 stride）。
Layout DeriveLayout(const at::Tensor& tensor) {
  if (tensor.dim() == 2 && tensor.size(0) > 1 && tensor.size(1) > 1 &&
      tensor.stride(0) == 1 && tensor.stride(1) == tensor.size(0)) {
    return Layout::kColumnMajor;
  }
  return Layout::kRowMajor;
}

// 返回张量底层存储数据指针；无存储或无数据指针返回 nullptr。
//
// 注意：meta/延迟张量的 TensorImpl 会在 ``storage()`` 中抛出“不可访问存储”
// 异常（storage_access_should_throw_）。校验视图只需要存储身份，不读取数据，
// 因此这里通过 ``unsafe_storage()`` 取存储并直接检查数据指针，避免异常。
const void* StorageKey(const at::Tensor& tensor) {
  if (!tensor.defined() || !tensor.has_storage()) {
    return nullptr;
  }
  const c10::Storage& storage = tensor.unsafeGetTensorImpl()->unsafe_storage();
  if (storage.unsafeGetStorageImpl() == nullptr) {
    return nullptr;
  }
  return storage.data();
}

// 把 Graph IR 模型配置转换为 Runtime 模型配置。
RuntimeModelConfig MakeRuntimeConfig(const ModelConfig& config) {
  RuntimeModelConfig runtime;
  runtime.model_family = config.model_family;
  runtime.num_layers = config.num_layers;
  runtime.hidden_size = config.hidden_size;
  runtime.intermediate_size = config.intermediate_size;
  runtime.num_attention_heads = config.num_attention_heads;
  runtime.num_kv_heads = config.num_kv_heads;
  runtime.head_dim = config.head_dim;
  runtime.vocab_size = config.vocab_size;
  runtime.max_context_length = config.max_context_length;
  runtime.norm_eps = config.norm_eps;
  runtime.activation = config.activation;
  runtime.tie_word_embeddings = config.tie_word_embeddings;
  runtime.dtype = ToDataType(config.dtype);
  runtime.rope = config.rope;
  runtime.extra = config.extra;
  return runtime;
}

}  // namespace

DataType ToDataType(at::ScalarType scalar_type) {
  switch (scalar_type) {
    case at::ScalarType::Float:
      return DataType::kFloat32;
    case at::ScalarType::Half:
      return DataType::kFloat16;
    case at::ScalarType::BFloat16:
      return DataType::kBFloat16;
    case at::ScalarType::Char:
      return DataType::kInt8;
    case at::ScalarType::Byte:
      return DataType::kUInt8;
    case at::ScalarType::Int:
      return DataType::kInt32;
    case at::ScalarType::Long:
      return DataType::kInt64;
    case at::ScalarType::Bool:
      return DataType::kBool;
    case at::ScalarType::Double:
      return DataType::kFloat64;
    default:
      return DataType::kUnknown;
  }
}

at::ScalarType ToScalarType(DataType data_type) {
  switch (data_type) {
    case DataType::kFloat32:
      return at::ScalarType::Float;
    case DataType::kFloat16:
      return at::ScalarType::Half;
    case DataType::kBFloat16:
      return at::ScalarType::BFloat16;
    case DataType::kInt8:
      return at::ScalarType::Char;
    case DataType::kUInt8:
      return at::ScalarType::Byte;
    case DataType::kInt32:
      return at::ScalarType::Int;
    case DataType::kInt64:
      return at::ScalarType::Long;
    case DataType::kBool:
      return at::ScalarType::Bool;
    case DataType::kFloat64:
      return at::ScalarType::Double;
    case DataType::kUnknown:
      return at::ScalarType::Undefined;
  }
  return at::ScalarType::Undefined;
}

const char* ScalarTypeName(at::ScalarType scalar_type) {
  return DataTypeName(ToDataType(scalar_type));
}

SymbolicShape ConcreteSymbolicShape(const at::Tensor& tensor) {
  SymbolicShape shape;
  if (!tensor.defined()) {
    return shape;
  }
  shape.dims.reserve(static_cast<size_t>(tensor.dim()));
  for (int64_t index = 0; index < tensor.dim(); ++index) {
    shape.dims.push_back(Dim::Constant(tensor.size(index)));
  }
  return shape;
}

ValidationTensor DescribeTensor(const at::Tensor& tensor,
                                const SymbolicShape& shape,
                                std::optional<QuantizationDesc> quantization) {
  ValidationTensor view;
  view.defined = tensor.defined();
  if (!view.defined) {
    return view;
  }
  view.dtype = ScalarTypeName(tensor.scalar_type());
  view.shape = shape.dims;
  view.stride.reserve(static_cast<size_t>(tensor.dim()));
  for (int64_t index = 0; index < tensor.dim(); ++index) {
    view.stride.push_back(tensor.stride(index));
  }
  view.layout = DeriveLayout(tensor);
  view.device =
      tensor.device().is_cpu() ? DeviceType::kHost : DeviceType::kDevice;
  const int64_t offset = tensor.storage_offset() * tensor.element_size();
  view.offset = offset > 0 ? static_cast<uint64_t>(offset) : 0;
  view.size_bytes = static_cast<uint64_t>(tensor.nbytes());
  view.requires_grad = tensor.requires_grad();
  view.storage_key = StorageKey(tensor);
  view.quantization = std::move(quantization);
  return view;
}

Status ToTensorDesc(const at::Tensor& tensor, const SymbolicShape& shape,
                    std::optional<QuantizationDesc> quantization,
                    TensorDesc* out) {
  if (out == nullptr) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "TensorDesc output must not be null");
  }
  if (!tensor.defined()) {
    return Status::Error(ErrorCode::kInvalidArgument, "tensor is undefined");
  }
  if (tensor.layout() != at::kStrided) {
    return Status::Error(ErrorCode::kUnsupportedLayout,
                         "only strided tensors can be converted to TensorDesc");
  }
  if (shape.Rank() != static_cast<size_t>(tensor.dim())) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "symbolic shape rank does not match tensor rank");
  }
  for (size_t index = 0; index < shape.Rank(); ++index) {
    if (shape.dims[index].value != tensor.size(static_cast<int64_t>(index))) {
      return Status::Error(
          ErrorCode::kInvalidArgument,
          "symbolic shape example value does not match tensor size");
    }
  }
  const DataType dtype = ToDataType(tensor.scalar_type());
  if (dtype == DataType::kUnknown) {
    return Status::Error(ErrorCode::kUnsupportedDtype,
                         "unsupported tensor scalar type");
  }
  ValidationTensor view = DescribeTensor(tensor, shape, quantization);
  if (view.layout == Layout::kColumnMajor && shape.Rank() != 2) {
    return Status::Error(ErrorCode::kUnsupportedLayout,
                         "column-major tensor must be 2-D");
  }
  out->dtype = dtype;
  out->shape = shape.dims;
  out->stride = std::move(view.stride);
  out->layout = view.layout;
  out->device = view.device;
  out->offset = view.offset;
  out->size_bytes = view.size_bytes;
  out->quantization = std::move(quantization);
  return Status::Ok();
}

TensorDesc ToTensorDesc(const at::Tensor& tensor, const SymbolicShape& shape,
                        std::optional<QuantizationDesc> quantization) {
  TensorDesc descriptor;
  const Status status =
      ToTensorDesc(tensor, shape, std::move(quantization), &descriptor);
  if (!status.ok()) {
    return TensorDesc{};
  }
  return descriptor;
}

Status ConvertGraphToRuntimeGraph(const Graph& graph,
                                  std::unique_ptr<RuntimeGraph>* out) {
  if (out == nullptr) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "RuntimeGraph output must not be null");
  }
  RuntimeGraphBuilder builder;
  Status status = builder.SetSchemaVersion(graph.schema_version_major(),
                                           graph.schema_version_minor());
  if (!status.ok()) {
    return status;
  }
  status = builder.SetFusionRulesVersion(graph.fusion_rules_version());
  if (!status.ok()) {
    return status;
  }
  status = builder.SetModelConfig(MakeRuntimeConfig(graph.model_config()));
  if (!status.ok()) {
    return status;
  }

  for (const Node& node : graph.nodes()) {
    RuntimeNode runtime_node;
    runtime_node.name = node.name;
    runtime_node.op_type = node.op_type;
    runtime_node.fused_ops = node.fused_ops;
    runtime_node.attributes = node.attributes;
    runtime_node.precision = ToDataType(node.precision);
    runtime_node.kernel_key = node.kernel_key;
    runtime_node.fusion_rule_id = node.fusion_rule_id;
    runtime_node.binding_name = node.binding_name;
    runtime_node.copy_kind = node.copy_kind;
    // 输入/输出边 ID 与 Graph 边插入顺序一致，下面按名称解析。
    for (const std::string& edge_name : node.input_edges) {
      const EdgeId edge_id = graph.FindEdge(edge_name);
      if (edge_id == kInvalidEdgeId) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "node references missing edge: " + edge_name);
      }
      runtime_node.input_edges.push_back(edge_id);
    }
    for (const std::string& edge_name : node.output_edges) {
      const EdgeId edge_id = graph.FindEdge(edge_name);
      if (edge_id == kInvalidEdgeId) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "node references missing edge: " + edge_name);
      }
      runtime_node.output_edges.push_back(edge_id);
    }
    if (node.host_tensor.has_value()) {
      TensorDesc host_desc;
      status = ToTensorDesc(*node.host_tensor,
                            ConcreteSymbolicShape(*node.host_tensor),
                            std::nullopt, &host_desc);
      if (!status.ok()) {
        return PrefixStatus(status, "host_tensor " + node.name);
      }
      runtime_node.host_tensor = std::move(host_desc);
    }
    NodeId runtime_node_id = kInvalidNodeId;
    status = builder.AddNode(std::move(runtime_node), &runtime_node_id);
    if (!status.ok()) {
      return status;
    }
    if (runtime_node_id != graph.FindNode(node.name)) {
      return Status::Error(ErrorCode::kInternal,
                           "node id changed during runtime conversion");
    }
  }

  for (const Edge& edge : graph.edges()) {
    RuntimeEdge runtime_edge;
    runtime_edge.name = edge.name;
    runtime_edge.quantization = edge.quantization;
    runtime_edge.producer = edge.producer;
    runtime_edge.consumers = edge.consumers;
    runtime_edge.is_weight = edge.is_weight;
    runtime_edge.weight_ref = edge.weight_ref;
    runtime_edge.arena_id = -1;
    runtime_edge.memory_offset = 0;
    status = ToTensorDesc(edge.tensor, edge.symbolic_shape, edge.quantization,
                          &runtime_edge.tensor);
    if (!status.ok()) {
      return PrefixStatus(status, "edge " + edge.name);
    }
    EdgeId runtime_edge_id = kInvalidEdgeId;
    status = builder.AddEdge(std::move(runtime_edge), &runtime_edge_id);
    if (!status.ok()) {
      return status;
    }
    if (runtime_edge_id != graph.FindEdge(edge.name)) {
      return Status::Error(ErrorCode::kInternal,
                           "edge id changed during runtime conversion");
    }
  }

  for (const WeightDesc& weight : graph.weights()) {
    RuntimeWeightDesc runtime_weight;
    runtime_weight.ref = weight.ref;
    runtime_weight.source_name = weight.source_name;
    runtime_weight.checksum = weight.checksum;
    status = ToTensorDesc(weight.tensor, ConcreteSymbolicShape(weight.tensor),
                          weight.quantization, &runtime_weight.tensor);
    if (!status.ok()) {
      return PrefixStatus(status, "weight " + weight.ref);
    }
    runtime_weight.data_bytes = runtime_weight.tensor.size_bytes;
    runtime_weight.data_offset = 0;
    status = builder.AddWeight(std::move(runtime_weight));
    if (!status.ok()) {
      return status;
    }
  }

  status = builder.SetInputNodes(graph.input_nodes());
  if (!status.ok()) {
    return status;
  }
  status = builder.SetOutputNodes(graph.output_nodes());
  if (!status.ok()) {
    return status;
  }
  return builder.Finalize(out);
}

#ifdef MOBILENET_GRAPH_USE_LIBTORCH
at::Tensor NormalizeForSerialization(const at::Tensor& tensor) {
  return tensor.detach().to(at::kCPU).contiguous();
}
#endif  // MOBILENET_GRAPH_USE_LIBTORCH

}  // namespace graph
}  // namespace mobilenet
