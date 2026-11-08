// 模块：Runtime 公共数据类型实现
//
// 职责：
//   实现 DataType / Layout / DeviceType 的名称与大小查询。
//
// 边界：
//   不包含任何张量存储或设备后端逻辑。

#include "Runtime/common/data_types.h"

namespace mobilenet {

const char* DataTypeName(DataType dtype) {
  switch (dtype) {
    case DataType::kUnknown:
      return "unknown";
    case DataType::kFloat32:
      return "float32";
    case DataType::kFloat16:
      return "float16";
    case DataType::kBFloat16:
      return "bfloat16";
    case DataType::kInt8:
      return "int8";
    case DataType::kUInt8:
      return "uint8";
    case DataType::kInt32:
      return "int32";
    case DataType::kInt64:
      return "int64";
    case DataType::kBool:
      return "bool";
    case DataType::kFloat64:
      return "float64";
  }
  return "unknown";
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

uint32_t DataTypeSize(DataType dtype) {
  switch (dtype) {
    case DataType::kFloat32:
      return 4;
    case DataType::kFloat16:
    case DataType::kBFloat16:
      return 2;
    case DataType::kInt8:
    case DataType::kUInt8:
    case DataType::kBool:
      return 1;
    case DataType::kInt32:
      return 4;
    case DataType::kInt64:
    case DataType::kFloat64:
      return 8;
    case DataType::kUnknown:
      return 0;
  }
  return 0;
}

}  // namespace mobilenet
