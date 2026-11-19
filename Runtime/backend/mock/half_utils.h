// 模块：Mock 后端半精度转换工具
//
// 职责：
//   提供 FP32↔half、bfloat16→half 的位级转换，供 CPU 参考 Kernel 与
//   ConvertParam 使用；转换规则为 IEEE 754 binary16，round-to-nearest-even。
//
// 边界：
//   - 只做标量转换，不处理向量化、批量或设备内存；
//   - 不依赖 OpenCL 的 half 类型或厂商头文件；
//   - NaN/Inf 按 IEEE 规则保留符号与指数。
//
// 依赖：
//   C++17 标准库。

#ifndef MOBILENET_RUNTIME_BACKEND_MOCK_HALF_UTILS_H_
#define MOBILENET_RUNTIME_BACKEND_MOCK_HALF_UTILS_H_

#include <cstdint>
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace mock {

// 把 float32 转换为 half 位模式。
//
// 参数：
//   value：待转换的 float32 值；允许 NaN/Inf。
// 返回：
//   IEEE 754 binary16 的 16 位位模式，采用 round-to-nearest-even；
//   上溢返回对应符号的 Inf，下溢按次正规数/零处理。
FP16 FloatToHalf(FP32 value);

// 把 half 位模式转换为 float32。
//
// 参数：
//   value：IEEE 754 binary16 的 16 位位模式。
// 返回：
//   对应的 float32 值；NaN/Inf 按位模式还原。
FP32 HalfToFloat(FP16 value);

// 把 bfloat16 位模式转换为 half 位模式。
//
// 参数：
//   value：bfloat16 的 16 位位模式。
// 返回：
//   对应的 half 位模式；bfloat16 的精度高于 half，转换会舍入。
FP16 BFloat16ToHalf(BF16 value);

}  // namespace mock
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_MOCK_HALF_UTILS_H_
