// 模块：Mock CPU Kernel 工厂
//
// 职责：
//   按 KernelDesc.kernel_key 创建对应的 CPU 参考 Kernel。
//
// 边界：
//   - 只认识 8 个已支持 key；未知 key 返回 kUnsupportedOp；
//   - 不校验 op attr/形状（由 OpEnv/OpSpec 负责）；
//   - 不拥有 Backend/Buffer。
//
// 依赖：
//   Runtime/backend/kernel.h、Runtime/common/status.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_BACKEND_MOCK_KERNELS_MOCK_KERNEL_FACTORY_H_
#define MOBILENET_RUNTIME_BACKEND_MOCK_KERNELS_MOCK_KERNEL_FACTORY_H_

#include <memory>

#include "Runtime/backend/kernel.h"
#include "Runtime/common/status.h"

namespace mobilenet {
namespace mock {

// 创建 Mock CPU Kernel。
//
// 参数：
//   desc：kernel 描述；kernel_key 决定实现；
//   kernel：输出参数，成功时写入非空 Kernel。
// 返回：
//   成功返回 kOk；未知 key 返回 kUnsupportedOp；kernel 为空返回
//   kInvalidArgument。
Status CreateMockKernel(const KernelDesc& desc,
                        std::unique_ptr<Kernel>* kernel);

}  // namespace mock
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_MOCK_KERNELS_MOCK_KERNEL_FACTORY_H_
