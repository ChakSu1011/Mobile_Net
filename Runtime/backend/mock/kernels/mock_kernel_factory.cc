// 模块：Mock CPU Kernel 工厂实现
//
// 职责：
//   实现 kernel_key 到 CPU Kernel 创建函数的分发。
//
// 边界：
//   未知 key 返回 kUnsupportedOp；不校验 op attr/形状。
//
// 依赖：
//   Runtime/backend/mock/kernels/mock_kernel_factory.h、
//   mock_kernel_creators.h。

#include "Runtime/backend/mock/kernels/mock_kernel_factory.h"

#include <string>

#include "Runtime/backend/mock/kernels/mock_kernel_creators.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace mock {

Status CreateMockKernel(const KernelDesc& desc,
                        std::unique_ptr<Kernel>* kernel) {
  if (kernel == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "CreateMockKernel: kernel output must not be null");
  }
  if (desc.kernel_key == "gather") {
    return CreateGatherKernel(kernel);
  }
  if (desc.kernel_key == "rms_norm") {
    return CreateRmsNormKernel(kernel);
  }
  if (desc.kernel_key == "linear") {
    return CreateLinearKernel(kernel);
  }
  if (desc.kernel_key == "linear_swi_glu") {
    return CreateLinearSwiGluKernel(kernel);
  }
  if (desc.kernel_key == "per_head_rmsnorm_rope") {
    return CreatePerHeadRmsNormRopeKernel(kernel);
  }
  if (desc.kernel_key == "mega_attention") {
    return CreateMegaAttentionKernel(kernel);
  }
  if (desc.kernel_key == "add") {
    return CreateAddKernel(kernel);
  }
  if (desc.kernel_key == "sample_argmax") {
    return CreateSampleKernel(kernel);
  }
  return MakeStatus(ErrorCode::kUnsupportedOp,
                    "CreateMockKernel: unsupported kernel_key: " +
                        desc.kernel_key);
}

}  // namespace mock
}  // namespace mobilenet
