// 模块：Backend 执行型 OpEnv 实现
//
// 职责：
//   实现 SetStepContext()：校验目标 OpEnv 类型并注入本步上下文。
//
// 边界：
//   不做调度、KV 分配或 plan 生成；不拥有 OpEnv/StepContext。
//
// 依赖：
//   Runtime/backend/exec_op_env.h、Runtime/common/status.h。

#include "Runtime/backend/exec_op_env.h"
#include "Utils/cpp/types.h"

namespace mobilenet {

Status SetStepContext(runtime_graph::OpEnv* env,
                      const StepContext* context) {
  if (env == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "SetStepContext: env must not be null");
  }
  auto* exec_env = dynamic_cast<ExecOpEnv*>(env);
  if (exec_env == nullptr) {
    return MakeStatus(
        ErrorCode::kInternal,
        "SetStepContext: OpEnv was not created by Backend OpEnvFactory");
  }
  exec_env->set_step_context(context);
  return Status::Ok();
}

}  // namespace mobilenet
