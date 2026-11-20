// 模块：OCL Add Kernel
//
// 职责：
//   实现逐元素加法 out=a+b 的 OpenCL Kernel 宿主包装与创建函数。
//
// 边界：
//   - P4.2 基线：元素/offset 8 对齐时 half8，否则标量；
//   - 不校验广播/attr 语义；只校验视图/形状；
//   - 性能调优在 P4.3/P4.5。
//
// 依赖：
//   Runtime/backend/ocl/ocl_kernel.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_ADD_ADD_KERNEL_H_
#define MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_ADD_ADD_KERNEL_H_

#include <memory>
#include <string>

#include "Runtime/backend/ocl/ocl_kernel.h"

namespace mobilenet {
namespace ocl {

// Add OCL Kernel。
class AddOclKernel final : public OclKernel {
 public:
  AddOclKernel(OclContext* context, OclProgramCache* programs,
               std::string kernel_key);

  ~AddOclKernel() override = default;

  AddOclKernel(const AddOclKernel&) = delete;
  AddOclKernel& operator=(const AddOclKernel&) = delete;

 private:
  Status Prepare(const runtime_graph::KernelContext& context) override;
  Status ExecuteImpl(const runtime_graph::KernelContext& context) override;
};

// 创建 Add OCL Kernel。
Status CreateAddOclKernel(OclContext* context, OclProgramCache* programs,
                          const std::string& kernel_key,
                          std::unique_ptr<Kernel>* kernel);

}  // namespace ocl
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_ADD_ADD_KERNEL_H_
