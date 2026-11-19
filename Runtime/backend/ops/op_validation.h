// 模块：算子形状/精度校验工具
//
// 职责：
//   为 8 个 OpSpec 提供统一的张量形状、精度、布局、字节数与 KernelDesc
//   组装工具，保证加载期校验的一致性与错误可定位。
//
// 边界：
//   - 只做静态校验，不读取/写入数据；
//   - 不做 shape 推导，不做隐式广播或类型转换；
//   - 不依赖设备或 Kernel 实现。
//
// 依赖：
//   Graph/include/tensor.h、shape.h、op.h、
//   Runtime/backend/kernel.h、kernel_types.h、Runtime/common/status.h。

#ifndef MOBILENET_RUNTIME_BACKEND_OPS_OP_VALIDATION_H_
#define MOBILENET_RUNTIME_BACKEND_OPS_OP_VALIDATION_H_

#include <cstdint>
#include <vector>

#include "Graph/include/op.h"
#include "Graph/include/shape.h"
#include "Graph/include/tensor.h"
#include "Runtime/backend/kernel.h"
#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {

// 返回 Shape 的逻辑维度；max_dim 非法时返回空。
std::vector<I64> ShapeToDims(const runtime_graph::Shape& shape);

// 返回张量 Storage 的精度对应的 DataType；未知精度返回 kUnknown。
DataType TensorDataType(const runtime_graph::Tensor& tensor);

// 校验张量的精度、布局、形状字节数与设备类别。
//
// 参数：
//   tensor：待校验张量；
//   precision：期望的 Graph 精度；
//   op_name：op 名，用于错误消息；
//   role：张量角色（如 "input[0]"），用于错误消息。
// 返回：
//   成功返回 kOk；精度不支持返回 kUnsupportedDtype；布局不支持返回
//   kUnsupportedLayout；形状/字节数/设备不匹配返回 kInvalidArgument。
Status CheckTensor(const runtime_graph::Tensor& tensor,
                   runtime_graph::Precision precision, const C8* op_name,
                   const C8* role);

// 校验张量秩。
//
// 参数：
//   tensor：待校验张量；
//   rank：期望秩；
//   op_name：op 名；
//   role：张量角色。
// 返回：
//   成功返回 kOk；秩不符返回 kInvalidArgument。
Status CheckRank(const runtime_graph::Tensor& tensor, Usize rank,
                 const C8* op_name, const C8* role);

// 校验两个维度序列的前 prefix_len 维一致。
//
// 参数：
//   first/second：维度序列；
//   prefix_len：比较前缀长度；
//   message：错误消息前缀。
// 返回：
//   成功返回 kOk；长度不足或不一致返回 kInvalidArgument。
Status CheckSamePrefix(const std::vector<I64>& first,
                       const std::vector<I64>& second,
                       Usize prefix_len, const C8* message);

// 组装 KernelDesc。
//
// 参数：
//   op：源 op；
//   kernel_key：后端实现键；
//   precision_tensor：决定计算精度的张量（通常为浮点输出/输入）；
//   desc：输出参数。
// 返回：
//   成功返回 kOk；精度不可映射返回 kUnsupportedDtype。
Status BuildKernelDesc(const runtime_graph::Op& op,
                       const C8* kernel_key,
                       const runtime_graph::Tensor& precision_tensor,
                       KernelDesc* desc);

}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OPS_OP_VALIDATION_H_
