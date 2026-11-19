// 模块：Input/Output 边界 OpEnv
//
// 职责：
//   Input op 执行主机到设备上传，Output op 执行设备到主机下载；主机
//   Storage 来自 Engine 每步写入的 attr["storage"]，设备 Storage 来自
//   图输入/输出 Tensor（Loader 绑定）。
//
// 边界：
//   - 不计算算子、不拥有 Buffer、不决定主机缓冲生命周期；
//   - 不做 layout/dtype 转换；主机与设备精度/字节数必须一致；
//   - 只处理 kInput/kOutput 两种 op。
//
// 依赖：
//   Runtime/backend/exec_op_env.h、kernel_context.h、backend.h、
//   Graph/include/attr.h、storage.h、tensor.h。

#ifndef MOBILENET_RUNTIME_BACKEND_OPS_IO_OP_ENV_H_
#define MOBILENET_RUNTIME_BACKEND_OPS_IO_OP_ENV_H_

#include <cstdint>

#include "Runtime/backend/exec_op_env.h"
#include "Runtime/backend/kernel_context.h"
#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {

class Backend;

// Input/Output 边界执行环境。
class IoOpEnv final : public ExecOpEnv {
 public:
  // 创建一个 I/O OpEnv。
  //
  // 参数：
  //   op：Input 或 Output op，生命周期由 Graph 保证；
  //   backend：后端实例，非拥有。
  IoOpEnv(runtime_graph::Op* op, Backend* backend);

  ~IoOpEnv() override = default;

  IoOpEnv(const IoOpEnv&) = delete;
  IoOpEnv& operator=(const IoOpEnv&) = delete;
  IoOpEnv(IoOpEnv&&) = delete;
  IoOpEnv& operator=(IoOpEnv&&) = delete;

  // 绑定设备侧视图。
  //
  // 参数：
  //   graph_env：图级共享环境，允许为空。
  // 返回：
  //   成功返回 kOk；op 类型非法或设备 Storage 未绑定返回错误码。
  Status Init(runtime_graph::GraphEnv* graph_env) override;

  // 执行 H2D 或 D2H。
  //
  // 返回：
  //   成功返回 kOk；attr["storage"] 缺失/非法或传输失败返回错误码。
  Status Forward() override;

  // I/O 节点不需要 workspace。
  Bool NeedTempData() const override { return false; }

  // 是否为 I/O 边界节点。
  Bool IsIoNode() const override { return true; }

 private:
  // 非拥有后端。
  Backend* backend_ = nullptr;

  // 设备侧输入/输出张量视图。
  runtime_graph::TensorView device_view_;

  // true 表示 Input（H2D），false 表示 Output（D2H）。
  Bool is_input_ = true;
};

}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OPS_IO_OP_ENV_H_
