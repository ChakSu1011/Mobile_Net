// 模块：Runtime Graph 基础类型实现
//
// 职责：
//   实现 Precision、Layout、DeviceType 的稳定名称与元素字节数查询。
//
// 边界：
//   只做纯函数映射，不读取全局状态，不依赖设备后端。
//
// 依赖：
//   types.h。

#include "types.h"

namespace mobilenet {
namespace runtime_graph {

const char* PrecisionName(Precision precision) {
  switch (precision) {
    case Precision::kUnknown:
      return "unknown";
    case Precision::kFloat64:
      return "float64";
    case Precision::kFloat32:
      return "float32";
    case Precision::kFloat16:
      return "float16";
    case Precision::kBFloat16:
      return "bfloat16";
    case Precision::kFloat8E4M3:
      return "float8_e4m3";
    case Precision::kFloat8E5M2:
      return "float8_e5m2";
    case Precision::kInt64:
      return "int64";
    case Precision::kInt32:
      return "int32";
    case Precision::kInt16:
      return "int16";
    case Precision::kInt8:
      return "int8";
    case Precision::kUInt8:
      return "uint8";
    case Precision::kBool:
      return "bool";
  }
  return "unknown";
}

int64_t PrecisionByteSize(Precision precision) {
  switch (precision) {
    case Precision::kUnknown:
      return 0;
    case Precision::kFloat64:
    case Precision::kInt64:
      return 8;
    case Precision::kFloat32:
    case Precision::kInt32:
      return 4;
    case Precision::kFloat16:
    case Precision::kBFloat16:
    case Precision::kInt16:
      return 2;
    case Precision::kFloat8E4M3:
    case Precision::kFloat8E5M2:
    case Precision::kInt8:
    case Precision::kUInt8:
    case Precision::kBool:
      return 1;
  }
  return 0;
}

const char* LayoutName(Layout layout) {
  switch (layout) {
    case Layout::kRowMajor:
      return "row_major";
    case Layout::kColumnMajor:
      return "column_major";
  }
  return "unknown";
}

const char* DeviceTypeName(DeviceType device) {
  switch (device) {
    case DeviceType::kUnknown:
      return "unknown";
    case DeviceType::kHost:
      return "host";
    case DeviceType::kDevice:
      return "device";
  }
  return "unknown";
}

}  // namespace runtime_graph
}  // namespace mobilenet
