// 模块：OCL MegaAttention Kernel
//
// 职责：
//   实现 causal GQA attention 的 OCL Kernel 宿主包装与创建函数。
//
// 边界：
//   - P4.3 基线：一个 work-item 处理一个 query，D ≤ 128；
//   - 只支持 merge_heads=true、has_output_projection=false、
//     use_kv_cache=false；
//   - 性能调优（分块 online softmax / GQA 复用）在 P4.5。
//
// 依赖：
//   Runtime/backend/ocl/ocl_kernel.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_MEGA_ATTENTION_MEGA_ATTENTION_KERNEL_H_
#define MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_MEGA_ATTENTION_MEGA_ATTENTION_KERNEL_H_

#include <memory>
#include <string>

#include "Runtime/backend/ocl/ocl_kernel.h"

namespace mobilenet {
namespace ocl {

// MegaAttention OCL Kernel。
class MegaAttentionOclKernel final : public OclKernel {
 public:
  MegaAttentionOclKernel(OclContext* context, OclProgramCache* programs,
                         std::string kernel_key);

  ~MegaAttentionOclKernel() override = default;

  MegaAttentionOclKernel(const MegaAttentionOclKernel&) = delete;
  MegaAttentionOclKernel& operator=(const MegaAttentionOclKernel&) = delete;

 private:
  Status Prepare(const runtime_graph::KernelContext& context) override;
  Status ExecuteImpl(const runtime_graph::KernelContext& context) override;
};

// 创建 MegaAttention OCL Kernel。
Status CreateMegaAttentionOclKernel(OclContext* context,
                                    OclProgramCache* programs,
                                    const std::string& kernel_key,
                                    std::unique_ptr<Kernel>* kernel);

// Paged MegaAttention OCL Kernel。
//
// 功能：
//   一次 Execute 内顺序下发 paged_kv_store 与 mega_attention_paged；
//   KV 池与元数据由 AttentionContext 提供。
class MegaAttentionPagedOclKernel final : public OclKernel {
 public:
  MegaAttentionPagedOclKernel(OclContext* context,
                              OclProgramCache* programs,
                              std::string kernel_key);

  ~MegaAttentionPagedOclKernel() override;

  MegaAttentionPagedOclKernel(const MegaAttentionPagedOclKernel&) = delete;
  MegaAttentionPagedOclKernel& operator=(
      const MegaAttentionPagedOclKernel&) = delete;

 private:
  Status Prepare(const runtime_graph::KernelContext& context) override;
  Status ExecuteImpl(const runtime_graph::KernelContext& context) override;

  cl_kernel store_kernel_ = nullptr;
  cl_kernel attention_kernel_ = nullptr;
};

// 创建 Paged MegaAttention OCL Kernel。
Status CreateMegaAttentionPagedOclKernel(OclContext* context,
                                         OclProgramCache* programs,
                                         const std::string& kernel_key,
                                         std::unique_ptr<Kernel>* kernel);

}  // namespace ocl
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OCL_KERNELS_MEGA_ATTENTION_MEGA_ATTENTION_KERNEL_H_
