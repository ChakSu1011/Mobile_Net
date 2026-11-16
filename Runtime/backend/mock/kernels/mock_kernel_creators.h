// 模块：Mock CPU Kernel 创建函数声明
//
// 职责：
//   声明 8 个算子各自的 Kernel 创建函数，供 mock_kernel_factory.cc 分发。
//
// 边界：
//   - 仅 Mock 后端内部使用；不属于公共契约；
//   - 创建函数不读取 KernelContext，不做数学计算。
//
// 依赖：
//   Runtime/backend/kernel.h、Runtime/common/status.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_BACKEND_MOCK_KERNELS_MOCK_KERNEL_CREATORS_H_
#define MOBILENET_RUNTIME_BACKEND_MOCK_KERNELS_MOCK_KERNEL_CREATORS_H_

#include <memory>

#include "Runtime/backend/kernel.h"
#include "Runtime/common/status.h"

namespace mobilenet {
namespace mock {

// 创建 Gather CPU Kernel；kernel 为空返回 kInvalidArgument。
Status CreateGatherKernel(std::unique_ptr<Kernel>* kernel);

// 创建 RMSNorm CPU Kernel；kernel 为空返回 kInvalidArgument。
Status CreateRmsNormKernel(std::unique_ptr<Kernel>* kernel);

// 创建 Linear CPU Kernel；kernel 为空返回 kInvalidArgument。
Status CreateLinearKernel(std::unique_ptr<Kernel>* kernel);

// 创建 LinearSwiGlu CPU Kernel；kernel 为空返回 kInvalidArgument。
Status CreateLinearSwiGluKernel(std::unique_ptr<Kernel>* kernel);

// 创建 PerHeadRMSNormRope CPU Kernel；kernel 为空返回 kInvalidArgument。
Status CreatePerHeadRmsNormRopeKernel(std::unique_ptr<Kernel>* kernel);

// 创建 MegaAttention CPU Kernel；kernel 为空返回 kInvalidArgument。
Status CreateMegaAttentionKernel(std::unique_ptr<Kernel>* kernel);

// 创建 Add CPU Kernel；kernel 为空返回 kInvalidArgument。
Status CreateAddKernel(std::unique_ptr<Kernel>* kernel);

// 创建 Argmax Sample CPU Kernel；kernel 为空返回 kInvalidArgument。
Status CreateSampleKernel(std::unique_ptr<Kernel>* kernel);

}  // namespace mock
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_MOCK_KERNELS_MOCK_KERNEL_CREATORS_H_
