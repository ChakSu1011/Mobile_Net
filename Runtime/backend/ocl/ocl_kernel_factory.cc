// 模块：OCL Kernel 工厂实现
//
// 职责：
//   实现 kernel_key 到算子 Kernel 创建函数的分发。
//
// 边界：
//   P4.2 实现 gather/add/sample_argmax；其余 key 返回 kUnsupportedOp。
//
// 依赖：
//   Runtime/backend/ocl/ocl_kernel_factory.h、
//   kernels/gather/gather_kernel.h、kernels/add/add_kernel.h、
//   kernels/sample/sample_kernel.h。

#include "Runtime/backend/ocl/ocl_kernel_factory.h"

#include <string>

#include "Runtime/backend/ocl/kernels/add/add_kernel.h"
#include "Runtime/backend/ocl/kernels/gather/gather_kernel.h"
#include "Runtime/backend/ocl/kernels/linear/linear_kernel.h"
#include "Runtime/backend/ocl/kernels/linear_per_head_rmsnorm_rope/linear_per_head_rmsnorm_rope_kernel.h"
#include "Runtime/backend/ocl/kernels/linear_swi_glu/linear_swi_glu_kernel.h"
#include "Runtime/backend/ocl/kernels/mega_attention/mega_attention_kernel.h"
#include "Runtime/backend/ocl/kernels/rms_norm/rms_norm_kernel.h"
#include "Runtime/backend/ocl/kernels/sample/sample_kernel.h"

namespace mobilenet {
namespace ocl {

// 按 kernel_key 分发到具体 OCL Kernel 创建函数；未知 key 返回 kUnsupportedOp。
Status CreateOclKernel(const KernelDesc& desc, OclContext* context,
                       OclProgramCache* programs,
                       std::unique_ptr<Kernel>* kernel) {
  if (kernel == nullptr || context == nullptr || programs == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "CreateOclKernel: kernel/context/programs must not be null");
  }
  if (desc.kernel_key == "gather") {
    return CreateGatherOclKernel(context, programs, desc.kernel_key,
                                 kernel);
  }
  if (desc.kernel_key == "add") {
    return CreateAddOclKernel(context, programs, desc.kernel_key, kernel);
  }
  if (desc.kernel_key == "sample_argmax") {
    return CreateSampleOclKernel(context, programs, desc.kernel_key,
                                 kernel);
  }
  if (desc.kernel_key == "rms_norm") {
    return CreateRmsNormOclKernel(context, programs, desc.kernel_key,
                                  kernel);
  }
  if (desc.kernel_key == "linear") {
    return CreateLinearOclKernel(context, programs, desc.kernel_key,
                                 kernel);
  }
  if (desc.kernel_key == "linear_swi_glu") {
    return CreateLinearSwiGluOclKernel(context, programs, desc.kernel_key,
                                       kernel);
  }
  if (desc.kernel_key == "per_head_rmsnorm_rope") {
    return CreatePerHeadRmsNormRopeOclKernel(
        context, programs, desc.kernel_key, kernel);
  }
  if (desc.kernel_key == "mega_attention") {
    return CreateMegaAttentionOclKernel(context, programs, desc.kernel_key,
                                        kernel);
  }
  if (desc.kernel_key == "mega_attention_paged") {
    return CreateMegaAttentionPagedOclKernel(
        context, programs, desc.kernel_key, kernel);
  }
  return MakeStatus(ErrorCode::kUnsupportedOp,
                    "[backend=ocl][stage=CreateKernel] kernel_key=" +
                        desc.kernel_key +
                        ": not implemented");
}

}  // namespace ocl
}  // namespace mobilenet
