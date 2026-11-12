// 模块：Runtime Graph OpEnv
//
// 职责：
//   定义挂在 Op 上的运行时执行环境抽象，以及 MegaAttention 的具体示例；
//   Engine 初始化阶段按 OpType 创建 OpEnv，Forward 阶段调用其执行入口。
//
// 边界：
//   - 不拥有 Op，不拥有 tensor，也不申请显存；
//   - 不直接依赖 GPU/OCL 算子库，具体 kernel 调用由派生类实现；
//   - 首版只提供接口与示例，不实现真实 MegaAttention kernel。
//
// 依赖：
//   Runtime/common/status.h、C++17 标准库。

#ifndef MOBILENET_GRAPH_FIXED_INCLUDE_RUNTIME_GRAPH_OP_ENV_H_
#define MOBILENET_GRAPH_FIXED_INCLUDE_RUNTIME_GRAPH_OP_ENV_H_

#include "Runtime/common/status.h"

namespace mobilenet {
namespace runtime_graph {

class GraphEnv;
class Op;

// OpEnv 表示挂在 op 上的运行时执行环境。
//
// 功能：
//   保存所属 Op 指针和 GraphEnv 指针，提供 Init、Forward、NeedTempData 三个
//   运行时入口。
//
// 边界：
//   - 生命周期由 Op 通过 unique_ptr 持有；
//   - op_ 由 Graph 持有，OpEnv 不负责释放；
//   - graph_env_ 由 Graph 持有，可能为空。
class OpEnv {
 public:
  // 创建执行环境。
  //
  // 参数：
  //   op：所属 op 指针，生命周期由 Graph 保证。
  explicit OpEnv(Op* op);

  virtual ~OpEnv() = default;

  OpEnv(const OpEnv&) = delete;
  OpEnv& operator=(const OpEnv&) = delete;
  OpEnv(OpEnv&&) = delete;
  OpEnv& operator=(OpEnv&&) = delete;

  // 初始化执行环境。
  //
  // 参数：
  //   graph_env：graph 级共享环境，允许为空，生命周期由 Graph 保证。
  // 返回：
  //   默认实现只保存指针并返回 OK；派生类可在此完成 kernel 资源初始化。
  virtual Status Init(GraphEnv* graph_env);

  // 执行 op。
  //
  // 返回：
  //   成功返回 OK；kernel 未实现或执行失败返回对应错误码。
  virtual Status Forward() = 0;

  // 返回 op 是否需要 temp data。
  //
  // 返回：
  //   true 表示 Manager 需要为该 op 预留临时 workspace。
  virtual bool NeedTempData() const = 0;

 protected:
  Op* op_ = nullptr;
  GraphEnv* graph_env_ = nullptr;
};

// MegaAttentionOpEnv 是 MegaAttention 的具体执行环境示例。
//
// 功能：
//   演示如何从 OpEnv 派生并接入 GraphEnv；首版 Forward 返回 UNSUPPORTED_OP，
//   表示真实 kernel 尚未接入。
//
// 边界：
//   - 不实现 attention 计算；
//   - NeedTempData 返回 true，表示 attention 路径需要临时 workspace。
class MegaAttentionOpEnv : public OpEnv {
 public:
  // 创建 MegaAttention 执行环境。
  //
  // 参数：
  //   op：所属 MegaAttention op，生命周期由 Graph 保证。
  explicit MegaAttentionOpEnv(Op* op);

  ~MegaAttentionOpEnv() override = default;

  // 执行 MegaAttention；首版返回 UNSUPPORTED_OP。
  Status Forward() override;

  // 返回 true，表示该 op 需要 temp data。
  bool NeedTempData() const override;
};

}  // namespace runtime_graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_FIXED_INCLUDE_RUNTIME_GRAPH_OP_ENV_H_
