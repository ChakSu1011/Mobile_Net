// 模块：通用计算 OpEnv
//
// 职责：
//   为所有融合算子提供统一的执行环境：Init 校验 spec/attr/形状并创建
//   Kernel；Forward 组装 KernelContext 并调用 Kernel::Execute。
//
// 边界：
//   - 不实现算子数学、不申请 Buffer、不做 H2D/D2H；
//   - 不拥有 Op/Backend/Kernel 之外的资源；
//   - Input/Output 边界节点由 IoOpEnv 处理。
//
// 依赖：
//   Runtime/backend/exec_op_env.h、kernel_context.h、ops/op_spec.h、Graph。

#ifndef MOBILENET_RUNTIME_BACKEND_OPS_KERNEL_OP_ENV_H_
#define MOBILENET_RUNTIME_BACKEND_OPS_KERNEL_OP_ENV_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "Runtime/backend/exec_op_env.h"
#include "Runtime/backend/kernel_context.h"
#include "Runtime/backend/ops/op_spec.h"
#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {

class Backend;

// 通用计算 OpEnv。
class KernelOpEnv final : public ExecOpEnv {
 public:
  // 创建一个计算 OpEnv。
  //
  // 参数：
  //   op：所属 op，生命周期由 Graph 保证；
  //   backend：后端实例，非拥有，生命周期由 Executor 保证；
  //   spec：算子规格，生命周期为静态常量。
  KernelOpEnv(runtime_graph::Op* op, Backend* backend,
              const OpSpec* spec);

  ~KernelOpEnv() override = default;

  KernelOpEnv(const KernelOpEnv&) = delete;
  KernelOpEnv& operator=(const KernelOpEnv&) = delete;
  KernelOpEnv(KernelOpEnv&&) = delete;
  KernelOpEnv& operator=(KernelOpEnv&&) = delete;

  // 校验算子契约、创建 Kernel 并缓存张量视图。
  //
  // 参数：
  //   graph_env：图级共享环境，允许为空；需要 workspace 时非空。
  // 返回：
  //   成功返回 kOk；attr/形状/精度失败返回具体错误码；Kernel 创建失败
  //   返回 kUnsupportedOp/kKernelError。
  Status Init(runtime_graph::GraphEnv* graph_env) override;

  // 组装 KernelContext 并执行一次节点计算。
  //
  // 返回：
  //   成功返回 kOk；未设置 StepContext 返回 kStateError；Kernel 失败时
  //   返回其错误码并带 op 名前缀。
  Status Forward() override;

  // 是否需要图级 workspace。
  Bool NeedTempData() const override;

 private:
  // 从 Tensor 构建执行视图。
  //
  // 参数：
  //   tensor：源/目标张量；
  //   role：角色名，用于错误消息；
  //   view：输出参数。
  // 返回：
  //   成功返回 kOk；未绑定设备/精度不支持返回错误码。
  Status BuildView(const runtime_graph::Tensor& tensor, const C8* role,
                   runtime_graph::TensorView* view) const;

  // 非拥有后端。
  Backend* backend_ = nullptr;

  // 静态算子规格。
  const OpSpec* spec_ = nullptr;

  // 由 Backend 创建、由本对象拥有的 Kernel。
  std::unique_ptr<Kernel> kernel_;

  // 输入/输出视图缓存。
  std::vector<runtime_graph::TensorView> inputs_;
  std::vector<runtime_graph::TensorView> outputs_;

  // workspace 视图；无 workspace 时 data 为空。
  runtime_graph::TensorView workspace_;
};

}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OPS_KERNEL_OP_ENV_H_
