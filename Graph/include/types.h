// 模块：Runtime Graph 基础类型
//
// 职责：
//   定义 runtime 静态图 IR 使用的 Precision、Layout、DeviceType 枚举，以及
//   Shape 定长数组的维度上限；同时提供枚举名称与元素字节数查询，供 Storage、
//   Graph、Python 绑定和测试共用。
//
// 边界：
//   - 不定义张量容器、Shape、Storage 或图结构；
//   - 不依赖 ATen/torch/c10，也不包含 GPU/OCL 算子库头文件；
//   - Precision 只覆盖 IR 与显存规划需要的类型，不承诺与后端 kernel
//     枚举一一对应；
//   - Layout 只表达稠密行列主序，不表达阻塞、分块或 swizzle 布局。
//
// 依赖：
//   仅依赖 C++17 标准库。

#ifndef MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_TYPES_H_
#define MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_TYPES_H_

#include <cstdint>

namespace mobilenet {
namespace runtime_graph {

// Shape 支持的最大维度数。
//
// 取值 8 覆盖当前 Transformer/CNN 模型；超过该上限的 shape 应在构图前拒绝。
inline constexpr int64_t kMaxShapeDim = 8;

// 张量元素精度。
//
// 功能：
//   描述 Storage 中元素的数据类型，供权重加载、显存规划与后端 kernel 选择使用。
//
// 边界：
//   - 不描述量化 scale/zero_point；量化参数必须由独立元数据承载；
//   - kUnknown 表示尚未推导或未声明，不能用于真实 kernel 调用。
enum class Precision : uint8_t {
  kUnknown = 0,
  kFloat64 = 1,
  kFloat32 = 2,
  kFloat16 = 3,
  kBFloat16 = 4,
  kFloat8E4M3 = 5,
  kFloat8E5M2 = 6,
  kInt64 = 7,
  kInt32 = 8,
  kInt16 = 9,
  kInt8 = 10,
  kUInt8 = 11,
  kBool = 12,
};

// 张量内存布局。
//
// 功能：
//   描述 Storage 的逻辑布局；kRowMajor 表示最后一维连续，kColumnMajor 表示
//   第一维连续。
//
// 边界：
//   不表达 padding、tiling 或 vendor 私有布局；这些信息由 Shape::pitch 与
//   后端 kernel 约定共同描述。
enum class Layout : uint8_t {
  kRowMajor = 0,
  kColumnMajor = 1,
};

// 张量数据地址所在设备类别。
//
// 功能：
//   区分 host 可见内存与设备内存，供权重 realize、输入输出绑定和 Uninit 使用。
//
// 边界：
//   不携带具体设备索引；设备索引由 Runtime 后端单独维护。
enum class DeviceType : uint8_t {
  kUnknown = 0,
  kHost = 1,
  kDevice = 2,
};

// 返回精度枚举的稳定名称。
//
// 参数：
//   precision：精度枚举值。
// 返回：
//   常量字符串，例如 "float16"；未知值返回 "unknown"。
const char* PrecisionName(Precision precision);

// 返回单个元素的字节数。
//
// 参数：
//   precision：精度枚举值。
// 返回：
//   字节数；kUnknown 返回 0。该函数不处理打包低位宽类型。
int64_t PrecisionByteSize(Precision precision);

// 返回布局枚举的稳定名称。
//
// 参数：
//   layout：布局枚举值。
// 返回：
//   常量字符串 "row_major" 或 "column_major"。
const char* LayoutName(Layout layout);

// 返回设备类别枚举的稳定名称。
//
// 参数：
//   device：设备类别枚举值。
// 返回：
//   常量字符串 "unknown"、"host" 或 "device"。
const char* DeviceTypeName(DeviceType device);

}  // namespace runtime_graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_TYPES_H_
