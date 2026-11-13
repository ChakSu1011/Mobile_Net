// 模块：Runtime 公共数据类型（DataType / Layout / DeviceType）
//
// 职责：
//   为 Graph、序列化格式与 Kernel 契约提供与 LibTorch 无关的
//   数据类型、布局与设备类型枚举，并提供稳定的名称与字节数查询。
//
// 边界：
//   - 这些枚举不用于 Graph IR 的张量属性；Graph IR 直接使用
//     at::ScalarType / at::Layout / at::Device。
//   - 本文件不定义 TensorDesc 结构，TensorDesc 在 runtime_graph_types.h。
//
// 依赖：
//   仅依赖 C++17 标准库。

#ifndef MOBILENET_RUNTIME_COMMON_DATA_TYPES_H_
#define MOBILENET_RUNTIME_COMMON_DATA_TYPES_H_

#include <cstdint>

namespace mobilenet {

// Runtime 张量数据类型。
//
// 取值与 LibTorch 的 at::ScalarType 保持一一映射；kUnknown 表示未设置。
enum class DataType {
  kUnknown = 0,
  kFloat32 = 1,
  kFloat16 = 2,
  kBFloat16 = 3,
  kInt8 = 4,
  kUInt8 = 5,
  kInt32 = 6,
  kInt64 = 7,
  kBool = 8,
  kFloat64 = 9,
};

// Runtime 张量内存布局。
enum class Layout {
  kRowMajor = 0,
  kColumnMajor = 1,
};

// Runtime 张量所在设备类别。
//
// 设计文档 V5/V17 只要求区分 Host/Device；具体设备索引由 Runtime 后端维护，
// TensorDesc 不携带 cuda/ocl 等具体设备号。
enum class DeviceType {
  kUnknown = 0,
  kHost = 1,
  kDevice = 2,
};

// 返回数据类型稳定名称，例如 "float16"。
//
// 参数：
//   dtype：数据类型。
// 返回：
//   常量字符串；未知类型返回 "unknown"。
const char* DataTypeName(DataType dtype);

// 返回布局稳定名称，例如 "row_major"。
const char* LayoutName(Layout layout);

// 返回设备类别稳定名称，例如 "host" / "device"。
const char* DeviceTypeName(DeviceType device);

// 返回单个元素的字节数。
//
// 参数：
//   dtype：数据类型。
// 返回：
//   字节数；kUnknown 返回 0。
uint32_t DataTypeSize(DataType dtype);

}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_COMMON_DATA_TYPES_H_
