// 模块：OCL LinearPerHeadRMSNormRope Kernel
//
// 职责：
//   实现 linear + per-head RMSNorm + NeoX RoPE 的 OCL Kernel 宿主包装
//   与创建函数。
//
// 边界：
//   - P4.3 基线：一 (row, head) 一 WG；lane 负责一个 head 维度；
//   - D ≤ 128；rope_layout=neox；只校验视图/形状，attr 由 OpSpec 校验；
//   - 性能调优（GEMM 分块/cos-sin 表）在 P4.5。
//
// 依赖：
//   Runtime/backend/ocl/ocl_kernel.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_LINEAR_PER_HEAD_RMSNORM_ROPE_LINEAR_PER_HEAD_RMSNORM_ROPE_KERNEL_H_
#define MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_LINEAR_PER_HEAD_RMSNORM_ROPE_LINEAR_PER_HEAD_RMSNORM_ROPE_KERNEL_H_

#include <memory>
#include <string>

#include "Runtime/backend/ocl/ocl_kernel.h"

namespace mobilenet {
namespace ocl {

// LinearPerHeadRMSNormRope OCL Kernel。
class PerHeadRmsNormRopeOclKernel final : public OclKernel {
 public:
  PerHeadRmsNormRopeOclKernel(OclContext* context,
                              OclProgramCache* programs,
                              std::string kernel_key);

  ~PerHeadRmsNormRopeOclKernel() override = default;

  PerHeadRmsNormRopeOclKernel(const PerHeadRmsNormRopeOclKernel&) = delete;
  PerHeadRmsNormRopeOclKernel& operator=(
      const PerHeadRmsNormRopeOclKernel&) = delete;

 private:
  Status Prepare(const runtime_graph::KernelContext& context) override;
  Status ExecuteImpl(const runtime_graph::KernelContext& context) override;
};

// 创建 LinearPerHeadRMSNormRope OCL Kernel。
Status CreatePerHeadRmsNormRopeOclKernel(
    OclContext* context, OclProgramCache* programs,
    const std::string& kernel_key, std::unique_ptr<Kernel>* kernel);

}  // namespace ocl
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_LINEAR_PER_HEAD_RMSNORM_ROPE_LINEAR_PER_HEAD_RMSNORM_ROPE_KERNEL_H_
