// 模块：Python <-> at::Tensor 桥接
//
// 职责：
//   在正常 LibTorch 构建中使用 pybind11 官方 type caster；在无 MSVC/LibTorch
//   链接环境的测试构建中，直接通过 CPython/THPVariable 接口在
//   torch.Tensor 与 at::Tensor 之间转换。
//
// 边界：
//   - 只做对象/元数据转换，不执行张量运算；
//   - 测试构建不链接 LibTorch，仅使用 torch 头文件与 Python 已加载的
//     torch_python.dll；
//   - 该桥接仅用于工具链/测试，不进入移动端运行时。
//
// 依赖：
//   pybind11；正常模式依赖 torch/extension.h；桥接模式依赖 ATen 头文件、
//   torch/csrc/autograd/python_variable.h 与 Windows API。

#ifndef MOBILENET_GRAPH_CPP_PYTHON_TORCH_BRIDGE_H_
#define MOBILENET_GRAPH_CPP_PYTHON_TORCH_BRIDGE_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <stdexcept>
#include <string>

#ifdef MOBILENET_GRAPH_HOST_TORCH_BRIDGE

#include <ATen/ATen.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/typeid.h>
#include <torch/csrc/autograd/python_variable.h>
#include <windows.h>

namespace mobilenet {
namespace graph {
namespace py = pybind11;

// 把 Python torch.Tensor 解包为 at::Tensor（共享 TensorImpl）。
inline at::Tensor TensorFromPython(py::handle handle) {
  const py::module_ torch = py::module_::import("torch");
  if (!py::isinstance(handle, torch.attr("Tensor"))) {
    throw py::type_error("expected torch.Tensor");
  }
  return THPVariable_Unpack(handle.ptr());
}

// 把 at::Tensor 包装为 Python torch.Tensor（共享 TensorImpl）。
using TphVariableWrapFn = PyObject* (*)(const at::TensorBase&);

// 动态解析 torch_python.dll 中的 THPVariable_Wrap 导出符号。
inline TphVariableWrapFn ResolveTensorWrap() {
  static TphVariableWrapFn wrap = []() -> TphVariableWrapFn {
    py::module_::import("torch");
    HMODULE module = GetModuleHandleA("torch_python.dll");
    if (module == nullptr) {
      throw std::runtime_error("torch_python.dll is not loaded");
    }
    const char* names[] = {
        "?THPVariable_Wrap@@YAPEAU_object@@AEBVTensorBase@at@@@Z",
        "?THPVariable_Wrap@@YAPEAU_object@@AEBVTensorBase@at@@PEAU_typeobject@@"
        "@Z",
    };
    for (const char* name : names) {
      FARPROC address = GetProcAddress(module, name);
      if (address != nullptr) {
        return reinterpret_cast<TphVariableWrapFn>(address);
      }
    }
    throw std::runtime_error("THPVariable_Wrap export not found");
  }();
  return wrap;
}

// 把 at::Tensor 转为 Python torch.Tensor；未定义张量转为 None。
inline py::object TensorToPython(const at::Tensor& tensor) {
  if (!tensor.defined()) {
    return py::none();
  }
  PyObject* object =
      ResolveTensorWrap()(static_cast<const at::TensorBase&>(tensor));
  if (object == nullptr) {
    throw py::error_already_set();
  }
  return py::reinterpret_steal<py::object>(object);
}

// 把 Python torch.dtype 转为 at::ScalarType；None 转为 Undefined。
inline at::ScalarType ScalarTypeFromPython(py::handle handle) {
  if (handle.is_none()) {
    return at::ScalarType::Undefined;
  }
  const py::module_ torch = py::module_::import("torch");
  const std::pair<const char*, at::ScalarType> mappings[] = {
      {"float32", at::ScalarType::Float},
      {"float16", at::ScalarType::Half},
      {"bfloat16", at::ScalarType::BFloat16},
      {"float64", at::ScalarType::Double},
      {"int8", at::ScalarType::Char},
      {"uint8", at::ScalarType::Byte},
      {"int32", at::ScalarType::Int},
      {"int64", at::ScalarType::Long},
      {"bool", at::ScalarType::Bool},
  };
  for (const auto& mapping : mappings) {
    if (handle.is(torch.attr(mapping.first))) {
      return mapping.second;
    }
  }
  throw py::type_error("unsupported torch.dtype");
}

// 把 at::ScalarType 转为 Python torch.dtype。
inline py::object ScalarTypeToPython(at::ScalarType scalar_type) {
  const py::module_ torch = py::module_::import("torch");
  switch (scalar_type) {
    case at::ScalarType::Float:
      return torch.attr("float32");
    case at::ScalarType::Half:
      return torch.attr("float16");
    case at::ScalarType::BFloat16:
      return torch.attr("bfloat16");
    case at::ScalarType::Double:
      return torch.attr("float64");
    case at::ScalarType::Char:
      return torch.attr("int8");
    case at::ScalarType::Byte:
      return torch.attr("uint8");
    case at::ScalarType::Int:
      return torch.attr("int32");
    case at::ScalarType::Long:
      return torch.attr("int64");
    case at::ScalarType::Bool:
      return torch.attr("bool");
    case at::ScalarType::Undefined:
    default:
      return py::none();
  }
}

// 可选张量转换。
inline std::optional<at::Tensor> OptionalTensorFromPython(py::handle handle) {
  if (handle.is_none()) {
    return std::nullopt;
  }
  return TensorFromPython(handle);
}

inline py::object OptionalTensorToPython(
    const std::optional<at::Tensor>& tensor) {
  return tensor.has_value() ? TensorToPython(*tensor) : py::none();
}

}  // namespace graph
}  // namespace mobilenet

#else  // MOBILENET_GRAPH_HOST_TORCH_BRIDGE

#include <torch/extension.h>

namespace mobilenet {
namespace graph {
namespace py = pybind11;

inline at::Tensor TensorFromPython(py::handle handle) {
  return handle.cast<at::Tensor>();
}

inline py::object TensorToPython(const at::Tensor& tensor) {
  if (!tensor.defined()) {
    return py::none();
  }
  return py::cast(tensor);
}

inline at::ScalarType ScalarTypeFromPython(py::handle handle) {
  if (handle.is_none()) {
    return at::ScalarType::Undefined;
  }
  return handle.cast<at::ScalarType>();
}

inline py::object ScalarTypeToPython(at::ScalarType scalar_type) {
  if (scalar_type == at::ScalarType::Undefined) {
    return py::none();
  }
  return py::cast(scalar_type);
}

inline std::optional<at::Tensor> OptionalTensorFromPython(py::handle handle) {
  if (handle.is_none()) {
    return std::nullopt;
  }
  return handle.cast<at::Tensor>();
}

inline py::object OptionalTensorToPython(
    const std::optional<at::Tensor>& tensor) {
  return tensor.has_value() ? py::cast(*tensor) : py::none();
}

}  // namespace graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_HOST_TORCH_BRIDGE

#endif  // MOBILENET_GRAPH_CPP_PYTHON_TORCH_BRIDGE_H_
