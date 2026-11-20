// 模块：OCL 权重精度转换工具
//
// 职责：
//   为 OclBackend::ConvertParam 提供 FP32/BF16→FP16 的主机位级转换。
//
// 边界：
//   - 只做标量转换；不处理 Buffer/设备；
//   - 与 mock/half_utils 规则一致；后续可抽公共工具；
//   - FP16 为 IEEE 754 binary16 位模式，RN-even。
//
// 依赖：
//   Utils/cpp/types.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_BACKEND_OCL_OCL_CONVERT_H_
#define MOBILENET_RUNTIME_BACKEND_OCL_OCL_CONVERT_H_

#include "Utils/cpp/types.h"

namespace mobilenet {
namespace ocl {

// float32 → FP16 位模式（RN-even）。
U16 FloatToHalf(FP32 value);

// FP16 位模式 → float32。
FP32 HalfToFloat(U16 value);

// bfloat16 位模式 → FP16 位模式。
U16 BFloat16ToHalf(U16 value);

}  // namespace ocl
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OCL_OCL_CONVERT_H_
