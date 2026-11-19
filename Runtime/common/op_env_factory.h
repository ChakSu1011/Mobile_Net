// 模块：Runtime OpEnv 工厂接口
//
// 职责：
//   定义 Loader 为每个计算 op 创建 OpEnv 的统一工厂接口；具体实现由
//   ExecBackend 提供，负责把 Op/Backend/GraphEnv 组合成可执行 OpEnv。
//
// 边界：
//   - 工厂由 Loader 内部创建与使用，不作为 Engine 成员，也不出现在 Engine
//     对外接口；
//   - 工厂不拥有 Op/Backend/GraphEnv，只借用；
//   - 工厂不决定调度、KV 或 Graph 拓扑；
//   - 创建失败必须保证不泄漏部分构造的 OpEnv/Kernel。
//
// 依赖：
//   Runtime/common/status.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_COMMON_OP_ENV_FACTORY_H_
#define MOBILENET_RUNTIME_COMMON_OP_ENV_FACTORY_H_

#include <memory>

#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {

class Backend;

namespace runtime_graph {
class Op;
class GraphEnv;
class OpEnv;
}  // namespace runtime_graph

namespace runtime {

// OpEnv 工厂。
//
// 功能：
//   Loader 在 GraphEnv 建立后，对 GetOrderlyOPList() 中的每个计算 op 调用
//   Create；Input/Output op 的 I/O OpEnv 也由该工厂创建。
//
// 边界：
//   - 仅初始化阶段单线程调用；
//   - 不负责把 OpEnv 挂到 Op 上（Loader 调 Op::SetOpEnv）；
//   - 返回的 OpEnv 由 Op 通过 unique_ptr 持有。
class OpEnvFactory {
 public:
  virtual ~OpEnvFactory() = default;

  // 为单个 op 创建 OpEnv。
  //
  // 参数：
  //   op：目标 op，生命周期由 Graph 保证；
  //   backend：后端实例，生命周期由 Engine 保证；
  //   graph_env：图级共享环境，生命周期由 Graph 保证；
  //   out：输出参数，成功时写入非空 unique_ptr。
  // 返回：
  //   成功返回 kOk；后端缺少算子/精度返回 kUnsupportedOp/
  //   kUnsupportedDtype；其它失败返回 kKernelError。
  virtual Status Create(runtime_graph::Op* op, Backend* backend,
                        runtime_graph::GraphEnv* graph_env,
                        std::unique_ptr<runtime_graph::OpEnv>* out) = 0;
};

// 创建默认 OpEnv 工厂。
//
// 返回：
//   非空工厂；由调用方（Engine）持有。实现方为 ExecBackend。
std::unique_ptr<OpEnvFactory> CreateDefaultOpEnvFactory();

}  // namespace runtime
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_COMMON_OP_ENV_FACTORY_H_
