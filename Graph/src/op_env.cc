// 模块：Runtime Graph OpEnv 实现
//
// 职责：
//   实现 OpEnv::Init 的默认行为与 MegaAttentionOpEnv 的占位执行入口。
//
// 边界：
//   不调用任何 GPU/OCL kernel，不申请显存。
//
// 依赖：
//   op_env.h、Runtime/common/status.h。

#include "op_env.h"

#include "Runtime/common/status.h"

namespace mobilenet {
namespace runtime_graph {

OpEnv::OpEnv(Op* op) : op_(op) {}

Status OpEnv::Init(GraphEnv* graph_env) {
  graph_env_ = graph_env;
  return Status::Ok();
}

MegaAttentionOpEnv::MegaAttentionOpEnv(Op* op) : OpEnv(op) {}

Status MegaAttentionOpEnv::Forward() {
  return MakeStatus(
      ErrorCode::kUnsupportedOp,
      "MegaAttentionOpEnv::Forward: MegaAttention kernel is not implemented");
}

bool MegaAttentionOpEnv::NeedTempData() const { return true; }

}  // namespace runtime_graph
}  // namespace mobilenet
