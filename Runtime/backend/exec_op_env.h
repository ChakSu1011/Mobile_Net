// 模块：Backend 执行型 OpEnv 与 StepContext
//
// 职责：
//   定义 ExecOpEnv 基类与 Executor 每步注入的 StepContext。OpEnv::Forward()
//   无参，Attention/Sample 等算子所需的 plan/KV/seq 元数据通过 StepContext
//   传入；Executor 在每步开始时设置、结束时清空。
//
// 边界：
//   - StepContext 由 Executor（或单算子测试驱动）构造，生命周期仅一步；
//   - ExecOpEnv 只被本模块工厂创建的 OpEnv 继承；Executor 通过
//     SetStepContext() 统一注入，不直接依赖具体 OpEnv 类型；
//   - 不改变 Graph::OpEnv 与 Runtime 统一接口；
//   - 不负责调度、KV 分配或 plan 生成。
//
// 依赖：
//   Graph/include/op_env.h；Runtime/backend/kernel_context.h；
//   Runtime/common/kv_view.h、plan.h、status.h；C++17 标准库。

#ifndef MOBILENET_RUNTIME_BACKEND_EXEC_OP_ENV_H_
#define MOBILENET_RUNTIME_BACKEND_EXEC_OP_ENV_H_

#include "Graph/include/op_env.h"
#include "Runtime/backend/kernel_context.h"
#include "Runtime/common/kv_view.h"
#include "Runtime/common/plan.h"
#include "Runtime/common/status.h"

namespace mobilenet {

class Backend;
class Stream;

// 单步执行上下文。
//
// 功能：
//   携带 Executor 当前 Step 的 plan、KV 视图、序列长度与 workspace，
//   供 OpEnv 组装 KernelContext。
//
// 边界：
//   - 所有指针均为非拥有；plan/kv_view/seq_lens 必须在整步内有效；
//   - 不拥有 Buffer 或工作区内存；
//   - 单算子测试可只填 backend/batch/packed_tokens/seq_lens，其余为空。
struct StepContext {
  // 后端实例；非拥有。
  Backend* backend = nullptr;

  // 执行流；为空表示默认流。
  Stream* stream = nullptr;

  // 本步执行计划；单算子测试可为空。
  const runtime::ExecutionPlan* plan = nullptr;

  // KV 只读视图；首期可为空。
  const runtime::KvCacheView* kv_view = nullptr;

  // 每 batch 行有效序列长度；长度至少为 batch，可为空。
  const int32_t* seq_lens = nullptr;

  // batch 行数。
  int32_t batch = 1;

  // 本步打包后的有效 token 数。
  int32_t packed_tokens = 0;

  // 图级临时工作区；无 workspace 时 data 为空。
  runtime_graph::TensorView workspace;

  // 是否处于 Prefill 阶段。
  bool is_prefill = true;
};

// 执行型 OpEnv。
//
// 功能：
//   在 Graph 的 OpEnv 基础上增加 StepContext 注入能力；所有由本模块工厂
//   创建的计算与 I/O OpEnv 都继承本类，Executor 用 SetStepContext() 统一
//   设置本步上下文。
//
// 边界：
//   - 不实现 Forward/NeedTempData，派生类必须实现；
//   - step_context_ 仅在同一 Step 内有效，步末必须清空；
//   - 不拥有 StepContext 或其内部指针。
class ExecOpEnv : public runtime_graph::OpEnv {
 public:
  // 创建执行环境。
  //
  // 参数：
  //   op：所属 op 指针，生命周期由 Graph 保证；测试可传空。
  explicit ExecOpEnv(runtime_graph::Op* op) : runtime_graph::OpEnv(op) {}

  ~ExecOpEnv() override = default;

  ExecOpEnv(const ExecOpEnv&) = delete;
  ExecOpEnv& operator=(const ExecOpEnv&) = delete;
  ExecOpEnv(ExecOpEnv&&) = delete;
  ExecOpEnv& operator=(ExecOpEnv&&) = delete;

  // 是否为 I/O 边界节点（Input/Output）。
  //
  // 返回：
  //   默认返回 false；IoOpEnv 返回 true。
  virtual bool IsIoNode() const { return false; }

  // 写入本步上下文。
  //
  // 参数：
  //   context：本步上下文，允许为空表示清空。
  void set_step_context(const StepContext* context) {
    step_context_ = context;
  }

 protected:
  // 本步上下文；由 Executor 在 Forward 前设置、步末清空。
  const StepContext* step_context_ = nullptr;
};

// 为 OpEnv 注入本步上下文。
//
// 功能：
//   校验 env 非空且是本模块创建的 ExecOpEnv，然后写入 context；
//   用于 Executor 在每步开始统一设置、步末传空清空。
//
// 参数：
//   env：目标 OpEnv，非空；必须由本模块 OpEnvFactory 创建；
//   context：本步上下文，允许为空表示清空。
// 返回：
//   成功返回 kOk；env 为空返回 kInvalidArgument；类型不符返回 kInternal。
Status SetStepContext(runtime_graph::OpEnv* env, const StepContext* context);

}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_EXEC_OP_ENV_H_
