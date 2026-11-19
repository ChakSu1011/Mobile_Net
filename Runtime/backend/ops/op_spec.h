// 模块：融合算子规格（OpSpec）
//
// 职责：
//   把每个融合算子的静态契约收敛为一个 OpSpec：op 类型、kernel_key、
//   workspace 需求、加载期校验函数与 KernelDesc 组装函数。
//
// 边界：
//   - 不执行算子数学、不创建 Kernel；
//   - 不读取模型包、不修改 Op/Tensor；
//   - 一个 OpType 对应一个 OpSpec；新增算子必须更新 op_registry。
//
// 依赖：
//   Graph/include/op.h、op_type.h、Runtime/backend/kernel.h。

#ifndef MOBILENET_RUNTIME_BACKEND_OPS_OP_SPEC_H_
#define MOBILENET_RUNTIME_BACKEND_OPS_OP_SPEC_H_

#include <cstdint>

#include "Graph/include/op.h"
#include "Graph/include/op_type.h"
#include "Runtime/backend/kernel.h"
#include "Utils/cpp/types.h"

namespace mobilenet {

// 单个融合算子的静态规格。
struct OpSpec {
  // 图节点类型。
  runtime_graph::OpType op_type = runtime_graph::OpType::kUnknown;

  // 后端实现键，例如 "rms_norm"。
  const C8* kernel_key = "";

  // 是否需要图级临时 workspace。
  Bool need_workspace = false;

  // workspace 字节数；need_workspace=false 时忽略。
  U64 workspace_bytes = 0;

  // 加载期校验 attr/形状/精度。
  //
  // 参数：
  //   op：目标 op；
  // 返回：
  //   成功返回 kOk；非法返回 kInvalidArgument/kUnsupportedOp/
  //   kUnsupportedDtype/kUnsupportedLayout。
  Status (*validate)(const runtime_graph::Op& op) = nullptr;

  // 组装 KernelDesc。
  //
  // 参数：
  //   op：目标 op；
  //   desc：输出参数；
  // 返回：
  //   成功返回 kOk；精度不支持返回 kUnsupportedDtype。
  Status (*build_kernel_desc)(const runtime_graph::Op& op,
                              KernelDesc* desc) = nullptr;
};

// 按 OpType 查找规格。
//
// 参数：
//   op_type：图节点类型。
// 返回：
//   找到返回非空指针；未支持的类型返回 nullptr。
const OpSpec* FindOpSpec(runtime_graph::OpType op_type);

}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OPS_OP_SPEC_H_
