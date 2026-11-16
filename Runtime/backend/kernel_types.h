// 模块：Backend/Kernel 边界类型映射
//
// 职责：
//   定义 Graph IR 精度（runtime_graph::Precision）与后端边界精度
//   （DataType）之间的显式转换；Backend/Kernel 公共头统一通过本文件
//   完成精度类型转换，避免在实现中散落 switch。
//
// 边界：
//   - 只做枚举映射，不定义 Kernel 接口、算子类型或后端接口；
//   - 不引入新枚举；算子类型统一使用 Graph 的 runtime_graph::OpType，
//     图侧布局/设备统一使用 runtime_graph::{Layout,DeviceType}；
//   - 不依赖 LibTorch、OpenCL 或其他厂商头文件。
//
// 依赖：
//   Graph/include/types.h、Runtime/common/data_types.h；C++17 标准库。

#ifndef MOBILENET_RUNTIME_BACKEND_KERNEL_TYPES_H_
#define MOBILENET_RUNTIME_BACKEND_KERNEL_TYPES_H_

#include "Graph/include/types.h"
#include "Runtime/common/data_types.h"

namespace mobilenet {

// 把 Graph IR 精度转换为后端边界精度。
//
// 参数：
//   precision：Graph Storage 中的精度枚举。
// 返回：
//   对应的 DataType；无对应类型的精度（Float8、Int16）返回
//   DataType::kUnknown，调用方应返回 kUnsupportedDtype。
DataType DataTypeFromPrecision(runtime_graph::Precision precision);

// 把后端边界精度转换为 Graph IR 精度。
//
// 参数：
//   dtype：Backend/Kernel 使用的精度枚举。
// 返回：
//   对应的 runtime_graph::Precision；kUnknown 或无法映射时返回
//   runtime_graph::Precision::kUnknown。
runtime_graph::Precision PrecisionFromDataType(DataType dtype);

// 判断后端边界精度是否可映射到 Graph IR 精度。
//
// 参数：
//   dtype：Backend/Kernel 使用的精度枚举。
// 返回：
//   dtype 非 kUnknown 且能映射到非 kUnknown 的 Precision 时返回 true。
bool IsSupportedDataType(DataType dtype);

}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_KERNEL_TYPES_H_
