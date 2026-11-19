// 模块：Mock CPU Kernel 公共工具
//
// 职责：
//   为 8 个 CPU 参考 Kernel 提供 Buffer 访问、half 读写、形状计算与
//   KernelContext 参数校验工具。
//
// 边界：
//   - 只服务于 Mock 后端；不暴露到公共 Backend/Kernel 契约；
//   - 不做形状推导、属性语义校验或错误恢复；
//   - 指针访问假设 TensorView.data 是 MockBuffer*，offset 已在视图内。
//
// 依赖：
//   Runtime/backend/kernel_context.h、mock_buffer.h、half_utils.h、
//   C++17 标准库。

#ifndef MOBILENET_RUNTIME_BACKEND_MOCK_KERNELS_KERNEL_COMMON_H_
#define MOBILENET_RUNTIME_BACKEND_MOCK_KERNELS_KERNEL_COMMON_H_

#include <cstdint>
#include <vector>

#include "Runtime/backend/kernel_context.h"
#include "Runtime/backend/mock/mock_buffer.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace mock {

// 从 TensorView 取 MockBuffer；空指针返回 nullptr。
const MockBuffer* GetBuffer(const runtime_graph::TensorView& view);

// 从 TensorView 取可写 MockBuffer；空指针返回 nullptr。
MockBuffer* GetMutableBuffer(const runtime_graph::TensorView& view);

// 返回 half 数据起始指针；buffer 为空时返回 nullptr。
const U16* HalfData(const runtime_graph::TensorView& view);

// 返回可写 half 数据起始指针；buffer 为空时返回 nullptr。
U16* MutableHalfData(const runtime_graph::TensorView& view);

// 返回 int64 数据起始指针；buffer 为空时返回 nullptr。
const I64* Int64Data(const runtime_graph::TensorView& view);

// 返回可写 int64 数据起始指针；buffer 为空时返回 nullptr。
I64* MutableInt64Data(const runtime_graph::TensorView& view);

// 返回 Shape 的有效维度；max_dim 非法时返回空。
std::vector<I64> Dims(const runtime_graph::Shape& shape);

// 返回元素总数；max_dim 为 0 时返回 1。
I64 NumElements(const runtime_graph::Shape& shape);

// 读取 half 张量第 index 个元素的 F32 值。
F32 ReadHalf(const runtime_graph::TensorView& view, U64 index);

// 把 F32 写入 half 张量第 index 个元素（RN-even）。
void WriteHalf(const runtime_graph::TensorView& view, U64 index,
               F32 value);

// 校验 KernelContext 至少包含 count 个输入。
Status RequireInputCount(const runtime_graph::KernelContext& context,
                         U32 count);

// 校验 KernelContext 至少包含 count 个输出。
Status RequireOutputCount(const runtime_graph::KernelContext& context,
                          U32 count);

// 校验 attrs 非空。
Status RequireAttrs(const runtime_graph::KernelContext& context);

}  // namespace mock
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_MOCK_KERNELS_KERNEL_COMMON_H_
