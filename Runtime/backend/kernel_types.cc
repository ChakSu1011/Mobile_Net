// 模块：Backend/Kernel 边界类型映射实现
//
// 职责：
//   实现 runtime_graph::Precision 与 DataType 的双向映射及支持性判断。
//
// 边界：
//   不包含 Kernel 执行逻辑、设备 API 或算子类型定义。
//
// 依赖：
//   Runtime/backend/kernel_types.h。

#include "Runtime/backend/kernel_types.h"

namespace mobilenet {

DataType DataTypeFromPrecision(runtime_graph::Precision precision) {
  switch (precision) {
    case runtime_graph::Precision::kFloat64:
      return DataType::kFloat64;
    case runtime_graph::Precision::kFloat32:
      return DataType::kFloat32;
    case runtime_graph::Precision::kFloat16:
      return DataType::kFloat16;
    case runtime_graph::Precision::kBFloat16:
      return DataType::kBFloat16;
    case runtime_graph::Precision::kFloat8E4M3:
    case runtime_graph::Precision::kFloat8E5M2:
      return DataType::kUnknown;
    case runtime_graph::Precision::kInt64:
      return DataType::kInt64;
    case runtime_graph::Precision::kInt32:
      return DataType::kInt32;
    case runtime_graph::Precision::kInt16:
      return DataType::kUnknown;
    case runtime_graph::Precision::kInt8:
      return DataType::kInt8;
    case runtime_graph::Precision::kUInt8:
      return DataType::kUInt8;
    case runtime_graph::Precision::kBool:
      return DataType::kBool;
    case runtime_graph::Precision::kUnknown:
      return DataType::kUnknown;
  }
  return DataType::kUnknown;
}

runtime_graph::Precision PrecisionFromDataType(DataType dtype) {
  switch (dtype) {
    case DataType::kFloat64:
      return runtime_graph::Precision::kFloat64;
    case DataType::kFloat32:
      return runtime_graph::Precision::kFloat32;
    case DataType::kFloat16:
      return runtime_graph::Precision::kFloat16;
    case DataType::kBFloat16:
      return runtime_graph::Precision::kBFloat16;
    case DataType::kInt64:
      return runtime_graph::Precision::kInt64;
    case DataType::kInt32:
      return runtime_graph::Precision::kInt32;
    case DataType::kInt8:
      return runtime_graph::Precision::kInt8;
    case DataType::kUInt8:
      return runtime_graph::Precision::kUInt8;
    case DataType::kBool:
      return runtime_graph::Precision::kBool;
    case DataType::kUnknown:
      return runtime_graph::Precision::kUnknown;
  }
  return runtime_graph::Precision::kUnknown;
}

bool IsSupportedDataType(DataType dtype) {
  return dtype != DataType::kUnknown &&
         PrecisionFromDataType(dtype) != runtime_graph::Precision::kUnknown;
}

}  // namespace mobilenet
