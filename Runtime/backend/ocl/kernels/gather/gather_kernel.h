// 模块：OCL Gather Kernel
//
// 职责：
//   实现 token embedding 查表的 OpenCL Kernel 宿主包装与创建函数。
//
// 边界：
//   - P4.2 只做基础实现；性能调优在 P4.3/P4.5；
//   - 不校验 axis/attr 语义（OpSpec 已校验）；只校验视图/形状；
//   - 越界 id 由上层输入合法性保证。
//
// 依赖：
//   Runtime/backend/ocl/ocl_kernel.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_GATHER_GATHER_KERNEL_H_
#define MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_GATHER_GATHER_KERNEL_H_

#include <memory>
#include <string>

#include "Runtime/backend/ocl/ocl_kernel.h"

namespace mobilenet {
namespace ocl {

// Gather OCL Kernel。
class GatherOclKernel final : public OclKernel {
 public:
  GatherOclKernel(OclContext* context, OclProgramCache* programs,
                  std::string kernel_key);

  ~GatherOclKernel() override = default;

  GatherOclKernel(const GatherOclKernel&) = delete;
  GatherOclKernel& operator=(const GatherOclKernel&) = delete;

 private:
  Status Prepare(const runtime_graph::KernelContext& context) override;
  Status ExecuteImpl(const runtime_graph::KernelContext& context) override;
};

// 创建 Gather OCL Kernel。
Status CreateGatherOclKernel(OclContext* context,
                             OclProgramCache* programs,
                             const std::string& kernel_key,
                             std::unique_ptr<Kernel>* kernel);

}  // namespace ocl
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_GATHER_GATHER_KERNEL_H_
