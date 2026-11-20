// 模块：OCL LinearSwiGlu Kernel
//
// 职责：
//   实现 silu(x·gate^T)*(x·up^T) 的 OCL Kernel 宿主包装与创建函数。
//
// 边界：
//   - P4.3 基线：每 work-item 计算一个输出元素，gate/up 双累加；
//   - 只支持 activation=silu、transpose_weight=true、无 bias；
//   - 性能调优在 P4.5。
//
// 依赖：
//   Runtime/backend/ocl/ocl_kernel.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_LINEAR_SWI_GLU_LINEAR_SWI_GLU_KERNEL_H_
#define MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_LINEAR_SWI_GLU_LINEAR_SWI_GLU_KERNEL_H_

#include <memory>
#include <string>

#include "Runtime/backend/ocl/ocl_kernel.h"

namespace mobilenet {
namespace ocl {

// LinearSwiGlu OCL Kernel。
class LinearSwiGluOclKernel final : public OclKernel {
 public:
  LinearSwiGluOclKernel(OclContext* context, OclProgramCache* programs,
                        std::string kernel_key);

  ~LinearSwiGluOclKernel() override = default;

  LinearSwiGluOclKernel(const LinearSwiGluOclKernel&) = delete;
  LinearSwiGluOclKernel& operator=(const LinearSwiGluOclKernel&) = delete;

 private:
  Status Prepare(const runtime_graph::KernelContext& context) override;
  Status ExecuteImpl(const runtime_graph::KernelContext& context) override;
};

// 创建 LinearSwiGlu OCL Kernel。
Status CreateLinearSwiGluOclKernel(OclContext* context,
                                   OclProgramCache* programs,
                                   const std::string& kernel_key,
                                   std::unique_ptr<Kernel>* kernel);

}  // namespace ocl
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_LINEAR_SWI_GLU_LINEAR_SWI_GLU_KERNEL_H_
