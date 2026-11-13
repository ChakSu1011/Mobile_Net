// 模块：Runtime Graph Python 绑定
//
// 职责：
//   通过 pybind11 把 runtime_graph::Graph、Tensor 和枚举暴露给 Python；
//   Python 侧只暴露构图接口、排序接口和 dump_ir 调试接口，不暴露 Op、
//   Storage、Shape 对象。
//
// 边界：
//   - 不重复实现构图逻辑，所有操作转发到 C++ Graph；
//   - 不把 C++ 指针所有权交给 Python：Tensor 句柄的生命周期由 Graph 持有，
//     调用方必须保证 Graph 比 Tensor 句柄存活更久；
//   - create_constant 只接收 CPU 数据指针，不读取指针内容；
//   - 不链接 torch/ATen，也不依赖 GPU/OCL 运行时。
//
// 依赖：
//   pybind11、runtime_graph/graph.h、runtime_graph/tensor.h、
//   runtime_graph/op_type.h、runtime_graph/storage.h。

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Runtime/common/status.h"
#include "runtime_graph/attr.h"
#include "runtime_graph/graph.h"
#include "runtime_graph/op.h"
#include "runtime_graph/op_type.h"
#include "runtime_graph/storage.h"
#include "runtime_graph/tensor.h"
#include "runtime_graph/types.h"

namespace mobilenet {
namespace runtime_graph {
namespace {

namespace py = pybind11;

// 把 Status 转换为 Python 异常。
//
// 参数：
//   status：C++ 操作状态。
// 返回：
//   无；失败时抛出 std::runtime_error，成功时直接返回。
void ThrowIfError(const Status& status) {
  if (!status.ok()) {
    throw std::runtime_error(status.message());
  }
}

// 把 Python 列表转换为 int64_t 数组。
//
// 参数：
//   value：Python list 或 tuple。
// 返回：
//   int64_t 数组。
// 异常：
//   元素不是 bool/int 时抛出 std::invalid_argument。
std::vector<int64_t> ToInt64Vector(const py::handle& value) {
  std::vector<int64_t> result;
  for (py::handle item : py::reinterpret_borrow<py::sequence>(value)) {
    if (!py::isinstance<py::int_>(item)) {
      throw std::invalid_argument("attribute array elements must be integers");
    }
    result.push_back(py::cast<int64_t>(item));
  }
  return result;
}

// 把 Python 列表转换为 double 数组。
//
// 参数：
//   value：Python list 或 tuple。
// 返回：
//   double 数组。
// 异常：
//   元素不是 int/float 时抛出 std::invalid_argument。
std::vector<double> ToDoubleVector(const py::handle& value) {
  std::vector<double> result;
  for (py::handle item : py::reinterpret_borrow<py::sequence>(value)) {
    if (!py::isinstance<py::int_>(item) &&
        !py::isinstance<py::float_>(item)) {
      throw std::invalid_argument(
          "attribute array elements must be numeric");
    }
    result.push_back(py::cast<double>(item));
  }
  return result;
}

// 把 Python 列表转换为字符串数组。
//
// 参数：
//   value：Python list 或 tuple。
// 返回：
//   字符串数组。
// 异常：
//   元素不是 str 时抛出 std::invalid_argument。
std::vector<std::string> ToStringVector(const py::handle& value) {
  std::vector<std::string> result;
  for (py::handle item : py::reinterpret_borrow<py::sequence>(value)) {
    if (!py::isinstance<py::str>(item)) {
      throw std::invalid_argument(
          "attribute array elements must be strings");
    }
    result.push_back(py::cast<std::string>(item));
  }
  return result;
}

// 把 Python 值转换为 AttrValue。
//
// 参数：
//   value：Python bool/int/float/str/list/tuple。
// 返回：
//   AttrValue；空 list 按 vector<int64_t> 处理。
// 异常：
//   类型不支持或数组元素类型混合时抛出 std::invalid_argument。
AttrValue ToAttrValue(const py::handle& value) {
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
  if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value)) {
    bool has_int = false;
    bool has_double = false;
    bool has_string = false;
    for (py::handle item : py::reinterpret_borrow<py::sequence>(value)) {
      if (py::isinstance<py::bool_>(item) || py::isinstance<py::int_>(item)) {
        has_int = true;
      } else if (py::isinstance<py::float_>(item)) {
        has_double = true;
      } else if (py::isinstance<py::str>(item)) {
        has_string = true;
      } else {
        throw std::invalid_argument(
            "attribute array contains unsupported element type");
      }
    }
    if (has_string && (has_int || has_double)) {
      throw std::invalid_argument(
          "attribute array cannot mix strings and numbers");
    }
    if (has_string) {
      return ToStringVector(value);
    }
    if (has_double) {
      return ToDoubleVector(value);
    }
    return ToInt64Vector(value);
  }
  throw std::invalid_argument(
      "attribute value must be bool, int, float, str, list, or tuple");
}

// 把 Python dict 转换为 AttrMap。
//
// 参数：
//   value：Python dict 或 None。
// 返回：
//   AttrMap；None 返回空表。
// 异常：
//   参数不是 dict、key 不是 str 或 value 类型不支持时抛出
//   std::invalid_argument。
AttrMap ToAttrMap(const py::object& value) {
  AttrMap result;
  if (value.is_none()) {
    return result;
  }
  if (!py::isinstance<py::dict>(value)) {
    throw std::invalid_argument("attr must be a dict or None");
  }
  const py::dict dict = value.cast<py::dict>();
  for (const auto& item : dict) {
    if (!py::isinstance<py::str>(item.first)) {
      throw std::invalid_argument("attribute names must be strings");
    }
    const std::string key = py::cast<std::string>(item.first);
    result.emplace(key, ToAttrValue(item.second));
  }
  return result;
}

// 由 Python 参数组装 constant Storage。
//
// 参数：
//   dims：逻辑维度；
//   pitch：物理维度；为空时复制 dims；
//   precision：元素精度；
//   layout：布局；
//   byte_size：storage 字节数；
//   data_ptr：CPU 数据地址，0 表示未设置。
// 返回：
//   Storage；device_type 固定为 kHost，reusable 默认 true，随后由
//   Graph::CreateConstant 置为 false。
// 异常：
//   维度数量超过 kMaxShapeDim、pitch 长度不匹配或维度为负时抛出
//   std::invalid_argument。
Storage MakeStorage(const std::vector<int64_t>& dims,
                    const std::vector<int64_t>& pitch, Precision precision,
                    Layout layout, uint64_t byte_size, uintptr_t data_ptr) {
  if (dims.size() > static_cast<size_t>(kMaxShapeDim)) {
    throw std::invalid_argument("dims size exceeds kMaxShapeDim");
  }
  if (!pitch.empty() && pitch.size() != dims.size()) {
    throw std::invalid_argument("pitch size must match dims size");
  }
  for (int64_t dim : dims) {
    if (dim < 0) {
      throw std::invalid_argument("dims must not contain negative values");
    }
  }
  for (int64_t value : pitch) {
    if (value < 0) {
      throw std::invalid_argument("pitch must not contain negative values");
    }
  }

  Storage storage;
  storage.shape.max_dim = static_cast<int64_t>(dims.size());
  for (size_t index = 0; index < dims.size(); ++index) {
    storage.shape.dims[index] = dims[index];
    storage.shape.pitch[index] = pitch.empty() ? dims[index] : pitch[index];
  }
  storage.precision = precision;
  storage.layout = layout;
  storage.device_type = DeviceType::kHost;
  storage.byte_size = static_cast<size_t>(byte_size);
  storage.data_ptr = reinterpret_cast<void*>(data_ptr);
  storage.reusable = true;
  return storage;
}

// 返回 tensor 名称列表的文本表示。
//
// 参数：
//   tensors：tensor 指针列表，允许包含空指针。
// 返回：
//   形如 "[a,b]" 的字符串；空指针输出 "<null>"。
std::string JoinTensorNames(const std::vector<Tensor*>& tensors) {
  std::ostringstream stream;
  stream << "[";
  for (size_t index = 0; index < tensors.size(); ++index) {
    if (index != 0) {
      stream << ",";
    }
    stream << (tensors[index] == nullptr ? "<null>" : tensors[index]->Name());
  }
  stream << "]";
  return stream.str();
}

// 生成只读 IR 文本。
//
// 参数：
//   graph：已完成构图并调用过 sort_op() 的 graph。
// 返回：
//   包含 inputs、constants、ops、outputs 四段的确定性文本；不输出真实
//   指针值，便于测试逐行断言。
std::string DumpIr(Graph& graph) {
  std::ostringstream stream;
  stream << "graph " << graph.Name() << "\n";

  stream << "inputs " << graph.GetAllInputOp().size() << "\n";
  for (Op* op : graph.GetAllInputOp()) {
    stream << "  " << op->Name() << " -> " << JoinTensorNames(op->DstTensors())
           << "\n";
  }

  stream << "constants " << graph.GetAllConstantTensor().size() << "\n";
  for (Tensor* tensor : graph.GetAllConstantTensor()) {
    stream << "  " << tensor->Name() << " "
           << StorageToString(tensor->GetStorage()) << "\n";
  }

  stream << "ops " << graph.GetOrderlyOPList().size() << "\n";
  const std::vector<Op*>& ordered_ops = graph.GetOrderlyOPList();
  for (size_t index = 0; index < ordered_ops.size(); ++index) {
    Op* op = ordered_ops[index];
    stream << "  [" << index << "] " << op->Name() << " "
           << OpTypeToStr(op->Type()) << " src="
           << JoinTensorNames(op->SrcTensors())
           << " dst=" << JoinTensorNames(op->DstTensors()) << " attrs={";
    bool first = true;
    for (const auto& entry : op->Attrs()) {
      if (!first) {
        stream << ",";
      }
      first = false;
      stream << entry.first << "=" << AttrValueToString(entry.second);
    }
    stream << "}\n";
  }

  stream << "outputs " << graph.GetAllOutputOp().size() << "\n";
  for (Op* op : graph.GetAllOutputOp()) {
    stream << "  " << op->Name() << " <- " << JoinTensorNames(op->SrcTensors())
           << "\n";
  }
  return stream.str();
}

}  // namespace

// 注册 runtime Graph Python 模块。
//
// 功能：
//   注册 Tensor、Graph、Precision、Layout、DeviceType、OpType，并绑定
//   create_op/create_constant/create_input/create_output/sort_op/dump_ir。
//
// 边界：
//   - 不暴露 Op、Storage、Shape 类型；
//   - Tensor 以非拥有引用返回，Graph 必须比 Tensor 句柄存活更久。
PYBIND11_MODULE(_graph, module) {
  module.doc() =
      "Runtime static graph IR (C++ Graph/Tensor with pybind11). Tensor "
      "handles are owned by Graph; keep the Graph alive while using them.";

  py::enum_<Precision>(module, "Precision")
      .value("kUnknown", Precision::kUnknown)
      .value("kFloat64", Precision::kFloat64)
      .value("kFloat32", Precision::kFloat32)
      .value("kFloat16", Precision::kFloat16)
      .value("kBFloat16", Precision::kBFloat16)
      .value("kFloat8E4M3", Precision::kFloat8E4M3)
      .value("kFloat8E5M2", Precision::kFloat8E5M2)
      .value("kInt64", Precision::kInt64)
      .value("kInt32", Precision::kInt32)
      .value("kInt16", Precision::kInt16)
      .value("kInt8", Precision::kInt8)
      .value("kUInt8", Precision::kUInt8)
      .value("kBool", Precision::kBool);

  py::enum_<Layout>(module, "Layout")
      .value("kRowMajor", Layout::kRowMajor)
      .value("kColumnMajor", Layout::kColumnMajor);

  py::enum_<DeviceType>(module, "DeviceType")
      .value("kUnknown", DeviceType::kUnknown)
      .value("kHost", DeviceType::kHost)
      .value("kDevice", DeviceType::kDevice);

  py::enum_<OpType>(module, "OpType")
      .value("kUnknown", OpType::kUnknown)
      .value("kInput", OpType::kInput)
      .value("kOutput", OpType::kOutput)
      .value("kMegaAttention", OpType::kMegaAttention)
      .value("kLinear", OpType::kLinear)
      .value("kLinearRmsNorm", OpType::kLinearRmsNorm)
      .value("kLinearRmsNormRope", OpType::kLinearRmsNormRope)
      .value("kLinearRmsNormRopeQuantPagedKVStore",
             OpType::kLinearRmsNormRopeQuantPagedKVStore)
      .value("kLinearRopeRotateQuant", OpType::kLinearRopeRotateQuant)
      .value("kLinearMul", OpType::kLinearMul)
      .value("kLinearApe", OpType::kLinearApe)
      .value("kRmsNorm", OpType::kRmsNorm)
      .value("kRmsNormRopeQuantPagedKVStore",
             OpType::kRmsNormRopeQuantPagedKVStore)
      .value("kRopeQuantLinear", OpType::kRopeQuantLinear)
      .value("kMegaMoe", OpType::kMegaMoe)
      .value("kDeepseekV3Router", OpType::kDeepseekV3Router)
      .value("kHashRouter", OpType::kHashRouter)
      .value("kScoreRouter", OpType::kScoreRouter)
      .value("kLinearSwiGlu", OpType::kLinearSwiGlu)
      .value("kGather", OpType::kGather)
      .value("kExpandEmbedding", OpType::kExpandEmbedding)
      .value("kAdd", OpType::kAdd)
      .value("kAddRmsNorm", OpType::kAddRmsNorm)
      .value("kFfn", OpType::kFfn)
      .value("kHcPre", OpType::kHcPre)
      .value("kHcPost", OpType::kHcPost)
      .value("kHcHead", OpType::kHcHead)
      .value("kRope", OpType::kRope)
      .value("kSparseMla", OpType::kSparseMla)
      .value("kQuant", OpType::kQuant)
      .value("kRotate", OpType::kRotate)
      .value("kKvCompress", OpType::kKvCompress)
      .value("kIndexScore", OpType::kIndexScore)
      .value("kTopK", OpType::kTopK)
      .value("kReshape", OpType::kReshape)
      .value("kPack", OpType::kPack)
      .value("kUnpack", OpType::kUnpack);

  py::class_<Tensor>(module, "Tensor",
                     "Non-owning handle to a tensor owned by Graph.")
      .def_property_readonly(
          "name", [](const Tensor& tensor) { return tensor.Name(); })
      .def_property_readonly("is_constant", &Tensor::IsConstant)
      .def_property_readonly("reusable", &Tensor::Reusable)
      .def("set_reusable", &Tensor::SetReusable, py::arg("reusable"));

  py::class_<Graph>(module, "Graph",
                    "Runtime static graph IR; owns ops, tensors and constants.")
      .def(py::init<const std::string&>(), py::arg("name"))
      .def(
          "create_op",
          [](Graph& graph, const std::string& name,
             const std::vector<Tensor*>& src_tensors, size_t out_cnt,
             OpType op_type, const py::object& attr) {
            const AttrMap attr_map = ToAttrMap(attr);
            std::vector<Tensor*> outputs;
            ThrowIfError(graph.CreateOp(name, src_tensors, out_cnt, op_type,
                                        attr_map, &outputs));
            py::list result;
            for (Tensor* tensor : outputs) {
              result.append(
                  py::cast(tensor, py::return_value_policy::reference));
            }
            return result;
          },
          py::arg("name"), py::arg("src_tensors"), py::arg("out_cnt"),
          py::arg("op_type"), py::arg("attr") = py::none())
      .def(
          "create_constant",
          [](Graph& graph, const std::string& name,
             const std::vector<int64_t>& dims,
             const std::vector<int64_t>& pitch, Precision precision,
             Layout layout, uint64_t byte_size, uintptr_t data_ptr) {
            const Storage storage = MakeStorage(dims, pitch, precision, layout,
                                                byte_size, data_ptr);
            Tensor* tensor = nullptr;
            ThrowIfError(graph.CreateConstant(name, storage, &tensor));
            return py::cast(tensor, py::return_value_policy::reference);
          },
          py::arg("name"), py::arg("dims"), py::arg("pitch"),
          py::arg("precision"), py::arg("layout"), py::arg("byte_size"),
          py::arg("data_ptr"))
      .def(
          "create_input",
          [](Graph& graph, const std::string& name) {
            Tensor* tensor = nullptr;
            ThrowIfError(graph.CreateInput(name, &tensor));
            return py::cast(tensor, py::return_value_policy::reference);
          },
          py::arg("name"))
      .def(
          "create_input",
          [](Graph& graph, const std::string& name, Precision precision,
             Layout layout) {
            Tensor* tensor = nullptr;
            ThrowIfError(
                graph.CreateInput(name, precision, layout, &tensor));
            return py::cast(tensor, py::return_value_policy::reference);
          },
          py::arg("name"), py::arg("precision"), py::arg("layout"))
      .def(
          "create_output",
          [](Graph& graph, const std::string& name, Tensor* input) {
            ThrowIfError(graph.CreateOutput(name, input));
          },
          py::arg("name"), py::arg("input"))
      .def("sort_op",
           [](Graph& graph) { ThrowIfError(graph.SortOp()); })
      .def("dump_ir", [](Graph& graph) { return DumpIr(graph); });
}

}  // namespace runtime_graph
}  // namespace mobilenet
