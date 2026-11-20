// 模块：OCL Sample Kernel
//
// 职责：
//   实现 argmax 采样的 OpenCL Kernel 宿主包装与创建函数。
//
// 边界：
//   - P4.2 基线：一行一 workgroup，标量扫描 + local 归约；
//   - 只支持 argmax、axis=-1；不校验 attr 语义（OpSpec 已校验）；
//   - 性能调优在后续阶段。
//
// 依赖：
//   Runtime/backend/ocl/ocl_kernel.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_SAMPLE_SAMPLE_KERNEL_H_
#define MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_SAMPLE_SAMPLE_KERNEL_H_

#include <memory>
#include <string>

#include "Runtime/backend/ocl/ocl_kernel.h"

namespace mobilenet {
namespace ocl {

// Sample OCL Kernel。
class SampleOclKernel final : public OclKernel {
 public:
  SampleOclKernel(OclContext* context, OclProgramCache* programs,
                  std::string kernel_key);

  ~SampleOclKernel() override = default;

  SampleOclKernel(const SampleOclKernel&) = delete;
  SampleOclKernel& operator=(const SampleOclKernel&) = delete;

 private:
  Status Prepare(const runtime_graph::KernelContext& context) override;
  Status ExecuteImpl(const runtime_graph::KernelContext& context) override;
};

// 创建 Sample OCL Kernel。
Status CreateSampleOclKernel(OclContext* context,
                             OclProgramCache* programs,
                             const std::string& kernel_key,
                             std::unique_ptr<Kernel>* kernel);

}  // namespace ocl
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_SAMPLE_SAMPLE_KERNEL_H_
