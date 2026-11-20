// 模块：OCL RMSNorm Kernel
//
// 职责：
//   实现最后一维 RMSNorm 的 OCL Kernel 宿主包装与创建函数。
//
// 边界：
//   - P4.3 基线：一行一 WG，FP32 平方和 + local 归约；
//   - 不校验 attr 语义（OpSpec 已校验）；只校验视图/形状；
//   - 性能调优在 P4.5。
//
// 依赖：
//   Runtime/backend/ocl/ocl_kernel.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_RMS_NORM_RMS_NORM_KERNEL_H_
#define MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_RMS_NORM_RMS_NORM_KERNEL_H_

#include <memory>
#include <string>

#include "Runtime/backend/ocl/ocl_kernel.h"

namespace mobilenet {
namespace ocl {

// RMSNorm OCL Kernel。
class RmsNormOclKernel final : public OclKernel {
 public:
  RmsNormOclKernel(OclContext* context, OclProgramCache* programs,
                   std::string kernel_key);

  ~RmsNormOclKernel() override = default;

  RmsNormOclKernel(const RmsNormOclKernel&) = delete;
  RmsNormOclKernel& operator=(const RmsNormOclKernel&) = delete;

 private:
  Status Prepare(const runtime_graph::KernelContext& context) override;
  Status ExecuteImpl(const runtime_graph::KernelContext& context) override;
};

// 创建 RMSNorm OCL Kernel。
Status CreateRmsNormOclKernel(OclContext* context,
                              OclProgramCache* programs,
                              const std::string& kernel_key,
                              std::unique_ptr<Kernel>* kernel);

}  // namespace ocl
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_RMS_NORM_RMS_NORM_KERNEL_H_
