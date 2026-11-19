// 模块：算子形状/精度校验工具实现
//
// 职责：
//   实现 Shape 转维度、Tensor 精度/布局/字节/设备校验与 KernelDesc 组装。
//
// 边界：
//   只做静态校验；不访问张量数据；不做隐式转换。
//
// 依赖：
//   Runtime/backend/ops/op_validation.h、kernel_types.h、
//   Runtime/common/data_types.h。

#include "Runtime/backend/ops/op_validation.h"

#include <string>

#include "Runtime/backend/kernel_types.h"
#include "Runtime/common/data_types.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace {

// 拼接带 op/role 的错误消息。
std::string TensorError(const C8* op_name, const C8* role,
                        const std::string& detail) {
  return std::string("op=") + (op_name != nullptr ? op_name : "?") +
         " " + (role != nullptr ? role : "tensor") + ": " + detail;
}

}  // namespace

std::vector<I64> ShapeToDims(const runtime_graph::Shape& shape) {
  if (shape.max_dim < 0 || shape.max_dim > runtime_graph::kMaxShapeDim) {
    return {};
  }
  std::vector<I64> dims;
  dims.reserve(static_cast<Usize>(shape.max_dim));
  for (I64 i = 0; i < shape.max_dim; ++i) {
    dims.push_back(shape.dims[static_cast<Usize>(i)]);
  }
  return dims;
}

DataType TensorDataType(const runtime_graph::Tensor& tensor) {
  return DataTypeFromPrecision(tensor.GetStorage().precision);
}

Status CheckTensor(const runtime_graph::Tensor& tensor,
                   runtime_graph::Precision precision, const C8* op_name,
                   const C8* role) {
  const runtime_graph::Storage& storage = tensor.GetStorage();
  if (storage.precision != precision) {
    return MakeStatus(ErrorCode::kUnsupportedDtype,
                      TensorError(op_name, role,
                                  "unexpected precision"));
  }
  if (storage.layout != runtime_graph::Layout::kRowMajor) {
    return MakeStatus(ErrorCode::kUnsupportedLayout,
                      TensorError(op_name, role,
                                  "only row_major is supported"));
  }
  if (storage.device_type != runtime_graph::DeviceType::kDevice) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      TensorError(op_name, role,
                                  "tensor is not bound to device"));
  }
  const DataType dtype = DataTypeFromPrecision(precision);
  const U32 element_size = DataTypeSize(dtype);
  if (element_size == 0) {
    return MakeStatus(ErrorCode::kUnsupportedDtype,
                      TensorError(op_name, role,
                                  "dtype has no element size"));
  }
  const std::vector<I64> dims = ShapeToDims(storage.shape);
  I64 elements = 1;
  for (const I64 dim : dims) {
    if (dim < 0) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        TensorError(op_name, role, "negative dimension"));
    }
    elements *= dim;
  }
  const U64 expected =
      static_cast<U64>(elements) * element_size;
  if (storage.byte_size != expected) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        TensorError(op_name, role, "byte_size does not match shape"));
  }
  return Status::Ok();
}

Status CheckRank(const runtime_graph::Tensor& tensor, Usize rank,
                 const C8* op_name, const C8* role) {
  const std::vector<I64> dims =
      ShapeToDims(tensor.GetStorage().shape);
  if (dims.size() != rank) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        TensorError(op_name, role, "unexpected tensor rank"));
  }
  return Status::Ok();
}

Status CheckSamePrefix(const std::vector<I64>& first,
                       const std::vector<I64>& second,
                       Usize prefix_len, const C8* message) {
  if (first.size() < prefix_len || second.size() < prefix_len) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      std::string(message != nullptr ? message : "shape") +
                          ": rank is too small");
  }
  for (Usize i = 0; i < prefix_len; ++i) {
    if (first[i] != second[i]) {
      return MakeStatus(
          ErrorCode::kInvalidArgument,
          std::string(message != nullptr ? message : "shape") +
              ": prefix mismatch");
    }
  }
  return Status::Ok();
}

Status BuildKernelDesc(const runtime_graph::Op& op,
                       const C8* kernel_key,
                       const runtime_graph::Tensor& precision_tensor,
                       KernelDesc* desc) {
  if (desc == nullptr || kernel_key == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "BuildKernelDesc: null desc or kernel_key");
  }
  const DataType dtype = TensorDataType(precision_tensor);
  if (dtype == DataType::kUnknown) {
    return MakeStatus(ErrorCode::kUnsupportedDtype,
                      "op=" + op.Name() +
                          " kernel precision is not supported");
  }
  desc->op_type = op.Type();
  desc->precision = dtype;
  desc->kernel_key = kernel_key;
  return Status::Ok();
}

}  // namespace mobilenet
