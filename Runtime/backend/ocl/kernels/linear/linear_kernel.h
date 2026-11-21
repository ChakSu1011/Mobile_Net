// 模块：OCL Linear Kernel
//
// 职责：
//   实现 y = x·W^T 的 NT GEMM OCL Kernel 宿主包装与创建函数。
//
// 边界：
//   - P4.3 基线：每 work-item 计算 4×4 输出块，FP32 累加；
//   - 只支持 transpose_weight=true、无 bias；
//   - 性能调优（更大分块/向量化）在 P4.5。
//
// 依赖：
//   Runtime/backend/ocl/ocl_kernel.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_LINEAR_LINEAR_KERNEL_H_
#define MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_LINEAR_LINEAR_KERNEL_H_

#include <memory>
#include <string>

#include "Runtime/backend/ocl/ocl_kernel.h"

namespace mobilenet {
namespace ocl {

// Linear OCL Kernel。
class LinearOclKernel final : public OclKernel {
 public:
  LinearOclKernel(OclContext* context, OclProgramCache* programs,
                  std::string kernel_key);

  ~LinearOclKernel() override;

  LinearOclKernel(const LinearOclKernel&) = delete;
  LinearOclKernel& operator=(const LinearOclKernel&) = delete;

 private:
  Status Prepare(const runtime_graph::KernelContext& context) override;
  Status ExecuteImpl(const runtime_graph::KernelContext& context) override;

  // true 表示该 Linear 是 LM Head，只计算 sample_rows 指定的行。
  Bool sample_rows_mode_ = false;

  // 与主 entry 并存的备用 tile（主为 4×8 时备用 4×4）。
  cl_kernel alt_kernel_ = nullptr;
};

// 创建 Linear OCL Kernel。
Status CreateLinearOclKernel(OclContext* context,
                             OclProgramCache* programs,
                             const std::string& kernel_key,
                             std::unique_ptr<Kernel>* kernel);

}  // namespace ocl
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_LINEAR_LINEAR_KERNEL_H_
