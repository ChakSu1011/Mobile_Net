// 模块：融合算子规格注册表
//
// 职责：
//   把 runtime_graph::OpType 映射到对应的 OpSpec；新增融合算子时必须
//   同步扩展本表与 BackendCapability.supported_ops。
//
// 边界：
//   - 只做静态查表，不创建 OpEnv/Kernel；
//   - 未支持的 OpType 返回 nullptr，由工厂转成 kUnsupportedOp；
//   - 不处理 Input/Output 边界节点（由 IoOpEnv 负责）。
//
// 依赖：
//   Runtime/backend/ops/op_spec.h、op_specs.h。

#include "Runtime/backend/ops/op_spec.h"

#include "Runtime/backend/ops/op_specs.h"

namespace mobilenet {

const OpSpec* FindOpSpec(runtime_graph::OpType op_type) {
  switch (op_type) {
    case runtime_graph::OpType::kGather:
      return &kGatherSpec;
    case runtime_graph::OpType::kRmsNorm:
      return &kRmsNormSpec;
    case runtime_graph::OpType::kLinear:
      return &kLinearSpec;
    case runtime_graph::OpType::kLinearSwiGlu:
      return &kLinearSwiGluSpec;
    case runtime_graph::OpType::kLinearPerHeadRMSNormRope:
      return &kPerHeadRmsNormRopeSpec;
    case runtime_graph::OpType::kMegaAttention:
      return &kMegaAttentionSpec;
    case runtime_graph::OpType::kAdd:
      return &kAddSpec;
    case runtime_graph::OpType::kSample:
      return &kSampleSpec;
    default:
      return nullptr;
  }
}

}  // namespace mobilenet
