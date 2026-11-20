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
#include "Runtime/backend/ocl/kernels/sample/sample_kernel.h"

namespace mobilenet {
namespace ocl {

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
  return MakeStatus(ErrorCode::kUnsupportedOp,
                    "[backend=ocl][stage=CreateKernel] kernel_key=" +
                        desc.kernel_key +
                        ": not implemented in P4.2");
}

}  // namespace ocl
}  // namespace mobilenet
