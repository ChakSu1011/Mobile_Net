// 模块：Graph IR 的 Python 绑定
//
// 职责：
//   将 Graph IR 的节点、边、权重、构建器、校验报告、融合规则表与公共枚举
//   暴露给 Python。Python 侧张量属性与精度属性分别暴露为 torch.Tensor 与
//   torch.dtype，不引入自定义张量/dtype/device 类型。
//
// 边界：
//   - 不暴露 RuntimeGraph.Forward；仅提供 Canonical JSON 与 TensorDesc 测试
//     辅助函数；
//   - 不定义第二套 Node/Edge 数据结构；
//   - 所有 C++ 异常在绑定边界转换为 GraphError。
//
// 依赖：
//   graph 公共/IR/Runtime 头文件、torch_bridge.h、qwen3_fixture.h、pybind11。

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Runtime/backend/backend.h"
#include "graph/fusion_rule.h"
#include "graph/graph.h"
#include "graph/graph_builder.h"
#include "graph/runtime_graph.h"
#include "graph/tensor_desc.h"
#include "graph/validation_common.h"
#include "qwen3_fixture.h"
#include "torch_bridge.h"

namespace mobilenet {
namespace graph {
namespace {

namespace py = pybind11;

// Python 侧图错误；携带错误码与最小定位对象。
class GraphError : public std::runtime_error {
 public:
  GraphError(ErrorCode code, std::string message, std::string node_name = "",
             std::string edge_name = "")
      : std::runtime_error(std::move(message)),
        code_(code),
        node_name_(std::move(node_name)),
        edge_name_(std::move(edge_name)) {}

  ErrorCode code() const { return code_; }
  const std::string& node_name() const { return node_name_; }
  const std::string& edge_name() const { return edge_name_; }
  std::string detail() const { return what(); }

 private:
  ErrorCode code_;
  std::string node_name_;
  std::string edge_name_;
};

// 状态失败时抛出 GraphError。
void CheckStatus(const Status& status, std::string node_name = "",
                 std::string edge_name = "") {
  if (!status.ok()) {
    throw GraphError(status.code(), status.message(), std::move(node_name),
                     std::move(edge_name));
  }
}

// 把 Python 属性值转换为 AttributeValue。
AttributeValue AttributeFromPython(py::handle value) {
  if (py::isinstance<py::bool_>(value)) {
    return py::cast<bool>(value);
  }
  if (py::isinstance<py::int_>(value)) {
    return py::cast<int64_t>(value);
  }
  if (py::isinstance<py::float_>(value)) {
    return py::cast<double>(value);
  }
  if (py::isinstance<py::str>(value)) {
    return py::cast<std::string>(value);
  }
  if (!py::isinstance<py::list>(value) && !py::isinstance<py::tuple>(value)) {
    throw py::type_error(
        "attribute value must be bool/int/float/str or homogeneous list");
  }
  const py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
  const size_t size = sequence.size();
  if (size == 0) {
    return std::vector<int64_t>{};
  }
  bool all_string = true;
  bool all_integer = true;
  bool all_numeric = true;
  for (size_t index = 0; index < size; ++index) {
    const py::object item = sequence[index];
    all_string = all_string && py::isinstance<py::str>(item);
    all_integer =
        all_integer && !py::isinstance<py::float_>(item) &&
        (py::isinstance<py::bool_>(item) || py::isinstance<py::int_>(item));
    all_numeric = all_numeric && (py::isinstance<py::bool_>(item) ||
                                  py::isinstance<py::int_>(item) ||
                                  py::isinstance<py::float_>(item));
  }
  if (all_string) {
    std::vector<std::string> values;
    values.reserve(size);
    for (size_t index = 0; index < size; ++index) {
      values.push_back(py::cast<std::string>(sequence[index]));
    }
    return values;
  }
  if (all_integer) {
    std::vector<int64_t> values;
    values.reserve(size);
    for (size_t index = 0; index < size; ++index) {
      values.push_back(py::cast<int64_t>(sequence[index]));
    }
    return values;
  }
  if (all_numeric) {
    std::vector<double> values;
    values.reserve(size);
    for (size_t index = 0; index < size; ++index) {
      values.push_back(py::cast<double>(sequence[index]));
    }
    return values;
  }
  throw py::type_error(
      "attribute list must be homogeneous string or numeric values");
}

// 把 Python dict 转换为属性映射。
std::map<std::string, AttributeValue> AttributesFromPython(py::handle value) {
  if (value.is_none()) {
    return {};
  }
  if (!py::isinstance<py::dict>(value)) {
    throw py::type_error("attributes must be a dict");
  }
  std::map<std::string, AttributeValue> attributes;
  for (const auto& item : py::reinterpret_borrow<py::dict>(value)) {
    const std::string key = py::cast<std::string>(item.first);
    if (key.empty()) {
      throw py::value_error("attribute key must not be empty");
    }
    attributes.emplace(key, AttributeFromPython(item.second));
  }
  return attributes;
}

// 把 AttributeValue 转换为 Python 原生对象。
py::object AttributeToPython(const AttributeValue& value) {
  struct Visitor {
    py::object operator()(bool boolean) const { return py::bool_(boolean); }

    py::object operator()(int64_t integer) const { return py::int_(integer); }

    py::object operator()(double number) const { return py::float_(number); }

    py::object operator()(const std::string& text) const {
      return py::str(text);
    }

    py::object operator()(const std::vector<int64_t>& values) const {
      py::list result;
      for (int64_t value : values) {
        result.append(py::int_(value));
      }
      return result;
    }

    py::object operator()(const std::vector<double>& values) const {
      py::list result;
      for (double value : values) {
        result.append(py::float_(value));
      }
      return result;
    }

    py::object operator()(const std::vector<std::string>& values) const {
      py::list result;
      for (const std::string& value : values) {
        result.append(py::str(value));
      }
      return result;
    }
  };
  return std::visit(Visitor{}, value);
}

// 把属性映射转换为 Python dict。
py::dict AttributesToPython(
    const std::map<std::string, AttributeValue>& attributes) {
  py::dict result;
  for (const auto& attribute : attributes) {
    result[py::str(attribute.first)] = AttributeToPython(attribute.second);
  }
  return result;
}

// 量化粒度字符串解析。
QuantGranularity GranularityFromName(std::string_view name) {
  if (name == "none") {
    return QuantGranularity::kNone;
  }
  if (name == "per_tensor") {
    return QuantGranularity::kPerTensor;
  }
  if (name == "per_channel") {
    return QuantGranularity::kPerChannel;
  }
  if (name == "per_group") {
    return QuantGranularity::kPerGroup;
  }
  throw py::value_error("unknown quantization granularity: " +
                        std::string(name));
}

// 可选量化描述转换。
std::optional<QuantizationDesc> OptionalQuantizationFromPython(
    py::handle value) {
  if (value.is_none()) {
    return std::nullopt;
  }
  return py::cast<QuantizationDesc>(value);
}

// 可选量化描述输出。
py::object OptionalQuantizationToPython(
    const std::optional<QuantizationDesc>& quantization) {
  if (!quantization.has_value()) {
    return py::none();
  }
  return py::cast(*quantization);
}

// 把维度列表转换为 Python 原生列表。
py::list DimsToPython(const std::vector<Dim>& dims) {
  py::list result;
  for (const Dim& dim : dims) {
    py::dict item;
    item["is_symbolic"] = dim.is_symbolic;
    item["value"] = dim.value;
    item["symbol"] = dim.symbol;
    result.append(item);
  }
  return result;
}

// 生成 to_tensor_desc 字典。
py::dict TensorDescToDict(const TensorDesc& descriptor) {
  py::dict result;
  result["dtype"] = DataTypeName(descriptor.dtype);
  result["shape"] = DimsToPython(descriptor.shape);
  result["stride"] = descriptor.stride;
  result["layout"] = LayoutName(descriptor.layout);
  result["device"] = DeviceTypeName(descriptor.device);
  result["offset"] = descriptor.offset;
  result["size_bytes"] = descriptor.size_bytes;
  result["quantization"] =
      OptionalQuantizationToPython(descriptor.quantization);
  return result;
}

// 把节点列表转换为名称列表。
py::list NodeNames(const Graph& graph, const std::vector<NodeId>& node_ids) {
  py::list result;
  for (NodeId node_id : node_ids) {
    const Node* node = graph.GetNode(node_id);
    result.append(node == nullptr ? py::str("") : py::str(node->name));
  }
  return result;
}

// 注册枚举常量。
void BindEnums(py::module_& module) {
  py::enum_<OpType>(module, "OpType")
      .value("kUnknown", OpType::kUnknown)
      .value("kInput", OpType::kInput)
      .value("kOutput", OpType::kOutput)
      .value("kEmbedding", OpType::kEmbedding)
      .value("kRmsNorm", OpType::kRmsNorm)
      .value("kQkvProjection", OpType::kQkvProjection)
      .value("kAttention", OpType::kAttention)
      .value("kResidualAdd", OpType::kResidualAdd)
      .value("kMlpGateUpAct", OpType::kMlpGateUpAct)
      .value("kMlpDownResidual", OpType::kMlpDownResidual)
      .value("kLmHead", OpType::kLmHead)
      .value("kSampling", OpType::kSampling)
      .value("kCast", OpType::kCast);

  py::enum_<CopyKind>(module, "CopyKind")
      .value("kNone", CopyKind::kNone)
      .value("kH2D", CopyKind::kH2D)
      .value("kD2H", CopyKind::kD2H);

  py::enum_<BlockStage>(module, "BlockStage")
      .value("kUnknown", BlockStage::kUnknown)
      .value("kStage1AttnNorm", BlockStage::kStage1AttnNorm)
      .value("kStage2Qkv", BlockStage::kStage2Qkv)
      .value("kStage3Attention", BlockStage::kStage3Attention)
      .value("kStage4AttnResidual", BlockStage::kStage4AttnResidual)
      .value("kStage5MlpNorm", BlockStage::kStage5MlpNorm)
      .value("kStage6MlpGateUp", BlockStage::kStage6MlpGateUp)
      .value("kStage7MlpDownResidual", BlockStage::kStage7MlpDownResidual)
      .value("kOutsideBlock", BlockStage::kOutsideBlock);

  py::enum_<RopeLayout>(module, "RopeLayout")
      .value("kNeox", RopeLayout::kNeox)
      .value("kGptJ", RopeLayout::kGptJ);

  py::enum_<QuantGranularity>(module, "QuantGranularity")
      .value("kNone", QuantGranularity::kNone)
      .value("kPerTensor", QuantGranularity::kPerTensor)
      .value("kPerChannel", QuantGranularity::kPerChannel)
      .value("kPerGroup", QuantGranularity::kPerGroup);

  py::enum_<ErrorCode>(module, "ErrorCode")
      .value("kOk", ErrorCode::kOk)
      .value("kInvalidArgument", ErrorCode::kInvalidArgument)
      .value("kStateError", ErrorCode::kStateError)
      .value("kUnsupportedOp", ErrorCode::kUnsupportedOp)
      .value("kVersionMismatch", ErrorCode::kVersionMismatch)
      .value("kBufferError", ErrorCode::kBufferError)
      .value("kDeviceError", ErrorCode::kDeviceError)
      .value("kKernelError", ErrorCode::kKernelError)
      .value("kInternal", ErrorCode::kInternal)
      .value("kUnsupportedDtype", ErrorCode::kUnsupportedDtype)
      .value("kUnsupportedLayout", ErrorCode::kUnsupportedLayout);
}

// 注册基础值类型。
void BindValueTypes(py::module_& module) {
  py::class_<Dim>(module, "Dim")
      .def(py::init<bool, int64_t, std::string>(),
           py::arg("is_symbolic") = false, py::arg("value") = 0,
           py::arg("symbol") = "")
      .def_static("constant", &Dim::Constant, py::arg("value"))
      .def_static("symbolic", &Dim::Symbolic, py::arg("symbol"),
                  py::arg("example_value"))
      .def_readwrite("is_symbolic", &Dim::is_symbolic)
      .def_readwrite("value", &Dim::value)
      .def_readwrite("symbol", &Dim::symbol)
      .def("__repr__", [](const Dim& dim) {
        return dim.is_symbolic
                   ? "Dim.symbolic('" + dim.symbol + "', " +
                         std::to_string(dim.value) + ")"
                   : "Dim.constant(" + std::to_string(dim.value) + ")";
      });

  py::class_<SymbolicShape>(module, "SymbolicShape")
      .def(py::init<>())
      .def(py::init<std::vector<Dim>>(), py::arg("dims"))
      .def_property_readonly("rank", &SymbolicShape::Rank)
      .def_property_readonly("is_fully_concrete",
                             &SymbolicShape::IsFullyConcrete)
      .def_property_readonly("example_shape", &SymbolicShape::ExampleShape)
      .def_readwrite("dims", &SymbolicShape::dims)
      .def("__repr__", [](const SymbolicShape& shape) {
        std::string text = "SymbolicShape([";
        for (size_t index = 0; index < shape.dims.size(); ++index) {
          if (index != 0) {
            text += ", ";
          }
          const Dim& dim = shape.dims[index];
          text += dim.is_symbolic ? dim.symbol + "=" + std::to_string(dim.value)
                                  : std::to_string(dim.value);
        }
        text += "])";
        return text;
      });

  py::class_<QuantizationDesc>(module, "QuantizationDesc")
      .def(py::init([](std::string storage_dtype, std::string granularity,
                       std::vector<double> scales,
                       std::vector<int64_t> zero_points, int32_t group_size) {
             QuantizationDesc descriptor;
             descriptor.storage_dtype = DataTypeFromName(storage_dtype);
             if (descriptor.storage_dtype == DataType::kUnknown) {
               throw py::value_error("unknown quantization storage dtype: " +
                                     storage_dtype);
             }
             descriptor.granularity = GranularityFromName(granularity);
             descriptor.scales = std::move(scales);
             descriptor.zero_points = std::move(zero_points);
             descriptor.group_size = group_size;
             return descriptor;
           }),
           py::arg("storage_dtype") = "int8",
           py::arg("granularity") = "per_tensor",
           py::arg("scales") = std::vector<double>{},
           py::arg("zero_points") = std::vector<int64_t>{},
           py::arg("group_size") = 0)
      .def_property(
          "storage_dtype",
          [](const QuantizationDesc& descriptor) {
            return std::string(DataTypeName(descriptor.storage_dtype));
          },
          [](QuantizationDesc& descriptor, const std::string& name) {
            const DataType dtype = DataTypeFromName(name);
            if (dtype == DataType::kUnknown) {
              throw py::value_error("unknown quantization storage dtype: " +
                                    name);
            }
            descriptor.storage_dtype = dtype;
          })
      .def_property(
          "granularity",
          [](const QuantizationDesc& descriptor) {
            return std::string(QuantGranularityName(descriptor.granularity));
          },
          [](QuantizationDesc& descriptor, const std::string& name) {
            descriptor.granularity = GranularityFromName(name);
          })
      .def_readwrite("scales", &QuantizationDesc::scales)
      .def_readwrite("zero_points", &QuantizationDesc::zero_points)
      .def_readwrite("group_size", &QuantizationDesc::group_size);

  py::class_<RopeConfig>(module, "RopeConfig")
      .def(py::init<>())
      .def_readwrite("theta", &RopeConfig::theta)
      .def_readwrite("layout", &RopeConfig::layout)
      .def_readwrite("scaling_type", &RopeConfig::scaling_type)
      .def_readwrite("scaling_factor", &RopeConfig::scaling_factor)
      .def_readwrite("original_max_position_embeddings",
                     &RopeConfig::original_max_position_embeddings);

  py::class_<ModelConfig>(module, "ModelConfig")
      .def(py::init([]() {
        ModelConfig config;
        config.dtype = at::ScalarType::Half;
        return config;
      }))
      .def_readwrite("model_family", &ModelConfig::model_family)
      .def_readwrite("num_layers", &ModelConfig::num_layers)
      .def_readwrite("hidden_size", &ModelConfig::hidden_size)
      .def_readwrite("intermediate_size", &ModelConfig::intermediate_size)
      .def_readwrite("num_attention_heads", &ModelConfig::num_attention_heads)
      .def_readwrite("num_kv_heads", &ModelConfig::num_kv_heads)
      .def_readwrite("head_dim", &ModelConfig::head_dim)
      .def_readwrite("vocab_size", &ModelConfig::vocab_size)
      .def_readwrite("max_context_length", &ModelConfig::max_context_length)
      .def_readwrite("norm_eps", &ModelConfig::norm_eps)
      .def_readwrite("activation", &ModelConfig::activation)
      .def_readwrite("tie_word_embeddings", &ModelConfig::tie_word_embeddings)
      .def_property(
          "dtype",
          [](const ModelConfig& config) {
            return ScalarTypeToPython(config.dtype);
          },
          [](ModelConfig& config, py::object dtype) {
            config.dtype = ScalarTypeFromPython(dtype);
          })
      .def_readwrite("rope", &ModelConfig::rope)
      .def_property(
          "extra",
          [](const ModelConfig& config) {
            return AttributesToPython(config.extra);
          },
          [](ModelConfig& config, py::object extra) {
            config.extra = AttributesFromPython(extra);
          });

  py::class_<FusedOp>(module, "FusedOp")
      .def(py::init([](std::string source_name, std::string op_kind,
                       py::object attributes) {
             FusedOp fused_op;
             fused_op.source_name = std::move(source_name);
             fused_op.op_kind = std::move(op_kind);
             fused_op.attributes = AttributesFromPython(attributes);
             return fused_op;
           }),
           py::arg("source_name"), py::arg("op_kind"),
           py::arg("attributes") = py::dict())
      .def_readwrite("source_name", &FusedOp::source_name)
      .def_readwrite("op_kind", &FusedOp::op_kind)
      .def_property_readonly("attributes", [](const FusedOp& fused_op) {
        return AttributesToPython(fused_op.attributes);
      });
}

// 注册节点、边、权重。
void BindGraphTypes(py::module_& module) {
  py::class_<Edge>(module, "Edge")
      .def(py::init([](std::string name, py::object tensor,
                       SymbolicShape symbolic_shape, py::object quantization,
                       bool is_weight, std::string weight_ref) {
             return Edge(std::move(name), TensorFromPython(tensor),
                         std::move(symbolic_shape),
                         OptionalQuantizationFromPython(quantization),
                         is_weight, std::move(weight_ref));
           }),
           py::arg("name"), py::arg("tensor"), py::arg("symbolic_shape"),
           py::arg("quantization") = py::none(), py::arg("is_weight") = false,
           py::arg("weight_ref") = "")
      .def_property_readonly("name", [](const Edge& edge) { return edge.name; })
      .def_property(
          "tensor",
          [](const Edge& edge) { return TensorToPython(edge.tensor); },
          [](Edge& edge, py::object tensor) {
            edge.tensor = TensorFromPython(tensor);
          })
      .def_readwrite("symbolic_shape", &Edge::symbolic_shape)
      .def_property(
          "quantization",
          [](const Edge& edge) {
            return OptionalQuantizationToPython(edge.quantization);
          },
          [](Edge& edge, py::object quantization) {
            edge.quantization = OptionalQuantizationFromPython(quantization);
          })
      .def_property_readonly("producer",
                             [](const Edge& edge) -> int64_t {
                               return edge.producer == kInvalidNodeId
                                          ? -1
                                          : static_cast<int64_t>(edge.producer);
                             })
      .def_property_readonly("consumers",
                             [](const Edge& edge) { return edge.consumers; })
      .def_readwrite("is_weight", &Edge::is_weight)
      .def_readwrite("weight_ref", &Edge::weight_ref);

  py::class_<WeightDesc>(module, "WeightDesc")
      .def(py::init([](std::string ref, std::string source_name,
                       py::object tensor, py::object quantization,
                       uint64_t checksum) {
             return WeightDesc(std::move(ref), std::move(source_name),
                               TensorFromPython(tensor),
                               OptionalQuantizationFromPython(quantization),
                               checksum);
           }),
           py::arg("ref"), py::arg("source_name"), py::arg("tensor"),
           py::arg("quantization") = py::none(), py::arg("checksum") = 0)
      .def_readwrite("ref", &WeightDesc::ref)
      .def_readwrite("source_name", &WeightDesc::source_name)
      .def_property(
          "tensor",
          [](const WeightDesc& weight) {
            return TensorToPython(weight.tensor);
          },
          [](WeightDesc& weight, py::object tensor) {
            weight.tensor = TensorFromPython(tensor);
          })
      .def_property(
          "quantization",
          [](const WeightDesc& weight) {
            return OptionalQuantizationToPython(weight.quantization);
          },
          [](WeightDesc& weight, py::object quantization) {
            weight.quantization = OptionalQuantizationFromPython(quantization);
          })
      .def_readwrite("checksum", &WeightDesc::checksum);

  py::class_<Node>(module, "Node")
      .def(py::init([](std::string name, OpType op_type,
                       std::vector<FusedOp> fused_ops, py::object attributes,
                       std::vector<std::string> input_edges,
                       std::vector<std::string> output_edges,
                       py::object precision, std::string kernel_key,
                       std::string fusion_rule_id, std::string binding_name,
                       CopyKind copy_kind, py::object host_tensor) {
             const at::ScalarType scalar_type =
                 precision.is_none() ? at::ScalarType::Half
                                     : ScalarTypeFromPython(precision);
             return Node(std::move(name), op_type, std::move(fused_ops),
                         AttributesFromPython(attributes),
                         std::move(input_edges), std::move(output_edges),
                         scalar_type, std::move(kernel_key),
                         std::move(fusion_rule_id), std::move(binding_name),
                         copy_kind, OptionalTensorFromPython(host_tensor));
           }),
           py::arg("name"), py::arg("op_type"),
           py::arg("fused_ops") = std::vector<FusedOp>{},
           py::arg("attributes") = py::dict(),
           py::arg("input_edges") = std::vector<std::string>{},
           py::arg("output_edges") = std::vector<std::string>{},
           py::arg("precision") = py::none(), py::arg("kernel_key") = "",
           py::arg("fusion_rule_id") = "", py::arg("binding_name") = "",
           py::arg("copy_kind") = CopyKind::kNone,
           py::arg("host_tensor") = py::none())
      .def_readwrite("name", &Node::name)
      .def_readwrite("op_type", &Node::op_type)
      .def_property_readonly("fused_ops",
                             [](const Node& node) { return node.fused_ops; })
      .def_property_readonly(
          "attributes",
          [](const Node& node) { return AttributesToPython(node.attributes); })
      .def_readwrite("input_edges", &Node::input_edges)
      .def_readwrite("output_edges", &Node::output_edges)
      .def_property(
          "precision",
          [](const Node& node) { return ScalarTypeToPython(node.precision); },
          [](Node& node, py::object precision) {
            node.precision = ScalarTypeFromPython(precision);
          })
      .def_readwrite("kernel_key", &Node::kernel_key)
      .def_readwrite("fusion_rule_id", &Node::fusion_rule_id)
      .def_readwrite("binding_name", &Node::binding_name)
      .def_readwrite("copy_kind", &Node::copy_kind)
      .def_property(
          "host_tensor",
          [](const Node& node) {
            return OptionalTensorToPython(node.host_tensor);
          },
          [](Node& node, py::object tensor) {
            node.host_tensor = OptionalTensorFromPython(tensor);
          });
}

// 注册校验报告与问题。
void BindValidation(py::module_& module) {
  py::class_<ValidationIssue>(module, "ValidationIssue")
      .def_property_readonly("code",
                             [](const ValidationIssue& issue) {
                               return std::string(ErrorCodeName(issue.code));
                             })
      .def_readonly("message", &ValidationIssue::message)
      .def_readonly("node_name", &ValidationIssue::node_name)
      .def_readonly("edge_name", &ValidationIssue::edge_name)
      .def_readonly("binding_name", &ValidationIssue::binding_name)
      .def_readonly("rule_id", &ValidationIssue::rule_id);

  py::class_<ValidationReport>(module, "ValidationReport")
      .def_property_readonly("ok", &ValidationReport::ok)
      .def_property_readonly(
          "issues",
          [](const ValidationReport& report) { return report.issues(); })
      .def("__len__", &ValidationReport::size)
      .def("to_string", &ValidationReport::ToString);
}

// 注册融合规则。
void BindFusionRules(py::module_& module) {
  py::class_<FusionRule>(module, "FusionRule")
      .def_readonly("rule_id", &FusionRule::rule_id)
      .def_readonly("model_family", &FusionRule::model_family)
      .def_readonly("stage", &FusionRule::stage)
      .def_readonly("op_type", &FusionRule::op_type)
      .def_readonly("required_op_kinds", &FusionRule::required_op_kinds)
      .def_property_readonly(
          "required_attributes",
          [](const FusionRule& rule) {
            return AttributesToPython(rule.required_attributes);
          })
      .def_readonly("kernel_key", &FusionRule::kernel_key);

  py::class_<FusionRuleTable>(module, "FusionRuleTable")
      .def_static(
          "instance",
          []() -> FusionRuleTable* {
            return const_cast<FusionRuleTable*>(&FusionRuleTable::Instance());
          },
          py::return_value_policy::reference)
      .def_property_readonly(
          "version",
          [](const FusionRuleTable& table) { return table.version(); })
      .def_property_readonly(
          "rules", [](const FusionRuleTable& table) { return table.rules(); })
      .def("find",
           [](const FusionRuleTable& table,
              const std::string& rule_id) -> py::object {
             const FusionRule* rule = table.Find(rule_id);
             return rule == nullptr ? py::none() : py::cast(*rule);
           })
      .def(
          "find_by_stage",
          [](const FusionRuleTable& table, const std::string& model_family,
             BlockStage stage,
             const std::vector<std::string>& op_kinds) -> py::object {
            const FusionRule* rule =
                table.FindByStage(model_family, stage, op_kinds);
            return rule == nullptr ? py::none() : py::cast(*rule);
          },
          py::arg("model_family"), py::arg("stage"), py::arg("op_kinds"));
}

// 注册 Graph IR。
void BindGraph(py::module_& module) {
  py::class_<Graph, std::unique_ptr<Graph>>(module, "Graph")
      .def_property_readonly("schema_version_major",
                             &Graph::schema_version_major)
      .def_property_readonly("schema_version_minor",
                             &Graph::schema_version_minor)
      .def_property_readonly("fusion_rules_version",
                             &Graph::fusion_rules_version)
      .def_property_readonly("model_config", &Graph::model_config)
      .def_property_readonly("nodes",
                             [](const Graph& graph) { return graph.nodes(); })
      .def_property_readonly("edges",
                             [](const Graph& graph) { return graph.edges(); })
      .def_property_readonly("weights",
                             [](const Graph& graph) { return graph.weights(); })
      .def_property_readonly("topo_order",
                             [](const Graph& graph) {
                               return NodeNames(graph, graph.topo_order());
                             })
      .def_property_readonly("input_nodes",
                             [](const Graph& graph) {
                               return NodeNames(graph, graph.input_nodes());
                             })
      .def_property_readonly("output_nodes",
                             [](const Graph& graph) {
                               return NodeNames(graph, graph.output_nodes());
                             })
      .def("find_node",
           [](const Graph& graph, const std::string& name) {
             const NodeId node_id = graph.FindNode(name);
             return node_id == kInvalidNodeId ? int64_t{-1}
                                              : static_cast<int64_t>(node_id);
           })
      .def("find_edge",
           [](const Graph& graph, const std::string& name) {
             const EdgeId edge_id = graph.FindEdge(name);
             return edge_id == kInvalidEdgeId ? int64_t{-1}
                                              : static_cast<int64_t>(edge_id);
           })
      .def("node",
           [](const Graph& graph, const std::string& name) -> py::object {
             const NodeId node_id = graph.FindNode(name);
             const Node* node = graph.GetNode(node_id);
             return node == nullptr ? py::none() : py::cast(*node);
           })
      .def("edge",
           [](const Graph& graph, const std::string& name) -> py::object {
             const EdgeId edge_id = graph.FindEdge(name);
             const Edge* edge = graph.GetEdge(edge_id);
             return edge == nullptr ? py::none() : py::cast(*edge);
           })
      .def("weight",
           [](const Graph& graph, const std::string& ref) -> py::object {
             const WeightDesc* weight = graph.FindWeight(ref);
             return weight == nullptr ? py::none() : py::cast(*weight);
           })
      .def("node_inputs",
           [](const Graph& graph, const std::string& name) {
             py::list tensors;
             const NodeId node_id = graph.FindNode(name);
             if (node_id == kInvalidNodeId) {
               return tensors;
             }
             for (const at::Tensor& tensor : graph.NodeInputs(node_id)) {
               tensors.append(TensorToPython(tensor));
             }
             return tensors;
           })
      .def("node_outputs",
           [](const Graph& graph, const std::string& name) {
             py::list tensors;
             const NodeId node_id = graph.FindNode(name);
             if (node_id == kInvalidNodeId) {
               return tensors;
             }
             for (const at::Tensor& tensor : graph.NodeOutputs(node_id)) {
               tensors.append(TensorToPython(tensor));
             }
             return tensors;
           })
      .def("validate",
           [](const Graph& graph) {
             ValidationReport report;
             const Status status = graph.Validate(&report);
             if (!status.ok() && report.ok()) {
               throw GraphError(status.code(), status.message());
             }
             return report;
           })
      .def("to_canonical_json", &Graph::ToCanonicalJson);
}

// 注册构建器。
void BindBuilder(py::module_& module) {
  py::class_<GraphBuilder>(module, "GraphBuilder")
      .def(py::init<>())
      .def(
          "add_node",
          [](GraphBuilder& builder, Node node) {
            const std::string name = node.name;
            NodeId node_id = kInvalidNodeId;
            CheckStatus(builder.AddNode(std::move(node), &node_id), name);
            return node_id;
          },
          py::arg("node"))
      .def(
          "add_edge",
          [](GraphBuilder& builder, Edge edge) {
            const std::string name = edge.name;
            EdgeId edge_id = kInvalidEdgeId;
            CheckStatus(builder.AddEdge(std::move(edge), &edge_id), "", name);
            return edge_id;
          },
          py::arg("edge"))
      .def(
          "add_weight",
          [](GraphBuilder& builder, WeightDesc weight) {
            const std::string ref = weight.ref;
            CheckStatus(builder.AddWeight(std::move(weight)), "", ref);
          },
          py::arg("weight"))
      .def(
          "set_model_config",
          [](GraphBuilder& builder, ModelConfig config) {
            CheckStatus(builder.SetModelConfig(std::move(config)));
          },
          py::arg("model_config"))
      .def(
          "set_schema_version",
          [](GraphBuilder& builder, uint32_t major, uint32_t minor) {
            CheckStatus(builder.SetSchemaVersion(major, minor));
          },
          py::arg("major"), py::arg("minor"))
      .def(
          "set_fusion_rules_version",
          [](GraphBuilder& builder, const std::string& version) {
            CheckStatus(builder.SetFusionRulesVersion(version));
          },
          py::arg("version"))
      .def(
          "set_input_nodes",
          [](GraphBuilder& builder, std::vector<NodeId> node_ids) {
            CheckStatus(builder.SetInputNodes(std::move(node_ids)));
          },
          py::arg("node_ids"))
      .def(
          "set_output_nodes",
          [](GraphBuilder& builder, std::vector<NodeId> node_ids) {
            CheckStatus(builder.SetOutputNodes(std::move(node_ids)));
          },
          py::arg("node_ids"))
      .def(
          "add_input_node",
          [](GraphBuilder& builder, NodeId node_id) {
            CheckStatus(builder.AddInputNode(node_id));
          },
          py::arg("node_id"))
      .def(
          "add_output_node",
          [](GraphBuilder& builder, NodeId node_id) {
            CheckStatus(builder.AddOutputNode(node_id));
          },
          py::arg("node_id"))
      .def(
          "finalize",
          [](GraphBuilder& builder) -> Graph* {
            std::unique_ptr<Graph> graph;
            const Status status = builder.Finalize(&graph);
            CheckStatus(status);
            return graph.release();
          },
          py::return_value_policy::take_ownership)
      .def_property_readonly("validation_report",
                             &GraphBuilder::validation_report);
}

// 注册测试辅助函数。
void BindHelpers(py::module_& module) {
  module.def(
      "to_tensor_desc",
      [](py::object tensor, SymbolicShape shape, py::object quantization) {
        TensorDesc descriptor;
        CheckStatus(ToTensorDesc(TensorFromPython(tensor), shape,
                                 OptionalQuantizationFromPython(quantization),
                                 &descriptor));
        return TensorDescToDict(descriptor);
      },
      py::arg("tensor"), py::arg("symbolic_shape"),
      py::arg("quantization") = py::none());

  module.def(
      "convert_graph_to_runtime_json",
      [](const Graph& graph) {
        std::unique_ptr<RuntimeGraph> runtime_graph;
        CheckStatus(ConvertGraphToRuntimeGraph(graph, &runtime_graph));
        return runtime_graph->ToCanonicalJson();
      },
      py::arg("graph"));

  module.def(
      "build_qwen3_reference_graph",
      []() -> Graph* {
        Status status;
        std::unique_ptr<Graph> graph = BuildQwen3ReferenceGraph(&status);
        CheckStatus(status);
        return graph.release();
      },
      py::return_value_policy::take_ownership);
}

// 模块级 GraphError 类型指针；由模块属性持有强引用。
PyObject* g_graph_error_type = nullptr;

// 注册 GraphError 及异常翻译器。
//
// 说明：当前 pybind11 版本的 ExceptionTranslator 是函数指针，因此使用
// captureless lambda + 模块级类型指针。
void RegisterGraphError(py::module_& module) {
  py::object graph_error_class =
      py::reinterpret_steal<py::object>(PyErr_NewException(
          "mobilenet_graph.GraphError", PyExc_RuntimeError, nullptr));
  g_graph_error_type = graph_error_class.ptr();
  module.attr("GraphError") = graph_error_class;
  py::register_exception_translator([](std::exception_ptr pointer) {
    if (!pointer) {
      return;
    }
    try {
      std::rethrow_exception(pointer);
    } catch (py::error_already_set& error) {
      error.restore();
    } catch (const GraphError& error) {
      py::object error_type =
          py::reinterpret_borrow<py::object>(g_graph_error_type);
      py::object instance = error_type(error.what());
      instance.attr("code") = std::string(ErrorCodeName(error.code()));
      instance.attr("node_name") = error.node_name();
      instance.attr("edge_name") = error.edge_name();
      instance.attr("detail") = error.detail();
      PyErr_SetObject(g_graph_error_type, instance.ptr());
    } catch (const std::exception& error) {
      PyErr_SetString(PyExc_RuntimeError, error.what());
    }
  });
}

}  // namespace
}  // namespace graph
}  // namespace mobilenet

// 模块入口。
PYBIND11_MODULE(_graph, module) {
  module.doc() =
      "MobileNet Graph IR Python binding (torch.Tensor / torch.dtype native)";

  mobilenet::graph::RegisterGraphError(module);
  mobilenet::graph::BindEnums(module);
  mobilenet::graph::BindValueTypes(module);
  mobilenet::graph::BindGraphTypes(module);
  mobilenet::graph::BindValidation(module);
  mobilenet::graph::BindFusionRules(module);
  mobilenet::graph::BindGraph(module);
  mobilenet::graph::BindBuilder(module);
  mobilenet::graph::BindHelpers(module);
}
