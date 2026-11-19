// 模块：融合算子规格声明
//
// 职责：
//   声明 8 个融合算子的 OpSpec 常量，供 op_registry.cc 统一登记。
//
// 边界：
//   - 只有声明；定义在各算子模块的 <op>_spec.cc；
//   - 不属于公共 Backend/Kernel 契约。
//
// 依赖：
//   Runtime/backend/ops/op_spec.h。

#ifndef MOBILENET_RUNTIME_BACKEND_OPS_OP_SPECS_H_
#define MOBILENET_RUNTIME_BACKEND_OPS_OP_SPECS_H_

#include "Runtime/backend/ops/op_spec.h"
#include "Utils/cpp/types.h"

namespace mobilenet {

extern const OpSpec kGatherSpec;
extern const OpSpec kRmsNormSpec;
extern const OpSpec kLinearSpec;
extern const OpSpec kLinearSwiGluSpec;
extern const OpSpec kPerHeadRmsNormRopeSpec;
extern const OpSpec kMegaAttentionSpec;
extern const OpSpec kAddSpec;
extern const OpSpec kSampleSpec;

}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OPS_OP_SPECS_H_
