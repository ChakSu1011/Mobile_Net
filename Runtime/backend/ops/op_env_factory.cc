// 模块：OpEnv 工厂实现
//
// 职责：
//   为 Loader 提供默认 OpEnvFactory：Input/Output 创建 IoOpEnv，计算 op
//   按 OpSpec 创建 KernelOpEnv；创建后立即调用 Init。
//
// 边界：
//   - 不把 OpEnv 挂到 Op 上（Loader 调 Op::SetOpEnv）；
//   - 不创建 Buffer/Kernel 之外的运行时资源；
//   - 未知 op 返回 kUnsupportedOp，并保证不泄漏已构造对象。
//
// 依赖：
//   Runtime/common/op_env_factory.h、ops/kernel_op_env.h、ops/io_op_env.h、
//   ops/op_spec.h、Graph/include/op.h、graph_env.h。

#include "Runtime/common/op_env_factory.h"

#include <memory>

#include "Graph/include/graph_env.h"
#include "Graph/include/op.h"
#include "Runtime/backend/backend.h"
#include "Runtime/backend/ops/io_op_env.h"
#include "Runtime/backend/ops/kernel_op_env.h"
#include "Runtime/backend/ops/op_spec.h"

namespace mobilenet {
namespace runtime {
namespace {

// 默认 OpEnv 工厂。
class DefaultOpEnvFactory final : public OpEnvFactory {
 public:
  // 为单个 op 创建并初始化 OpEnv。
  //
  // 参数：
  //   op：目标 op；
  //   backend：非拥有后端；
  //   graph_env：图级共享环境，可为空；
  //   out：输出参数。
  // 返回：
  //   成功返回 kOk；未知 op 返回 kUnsupportedOp；Init 失败透传其错误码。
  Status Create(runtime_graph::Op* op, Backend* backend,
                runtime_graph::GraphEnv* graph_env,
                std::unique_ptr<runtime_graph::OpEnv>* out) override {
    if (op == nullptr || backend == nullptr || out == nullptr) {
      return MakeStatus(
          ErrorCode::kInvalidArgument,
          "OpEnvFactory::Create: op/backend/out must not be null");
    }

    std::unique_ptr<runtime_graph::OpEnv> op_env;
    if (op->Type() == runtime_graph::OpType::kInput ||
        op->Type() == runtime_graph::OpType::kOutput) {
      op_env = std::make_unique<IoOpEnv>(op, backend);
    } else {
      const OpSpec* spec = FindOpSpec(op->Type());
      if (spec == nullptr) {
        return MakeStatus(ErrorCode::kUnsupportedOp,
                          "op=" + op->Name() +
                              ": no OpSpec registered");
      }
      op_env = std::make_unique<KernelOpEnv>(op, backend, spec);
    }

    const Status status = op_env->Init(graph_env);
    if (!status.ok()) {
      return status;
    }
    *out = std::move(op_env);
    return Status::Ok();
  }
};

}  // namespace

std::unique_ptr<OpEnvFactory> CreateDefaultOpEnvFactory() {
  return std::make_unique<DefaultOpEnvFactory>();
}

}  // namespace runtime
}  // namespace mobilenet
