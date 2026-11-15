// 模块：Runtime Executor 统一接口
//
// 职责：
//   定义 Engine 调用的 IExecutor：消费 ExecutionPlan/KvCacheView，绑定主机
//   输入输出，按 Graph::GetOrderlyOPList() 驱动 OpEnv::Forward，回读
//   sample 输出并返回 StepResult。
//
// 边界：
//   - 不做调度、KV 分配或停止判定；
//   - 不释放 KV、不调用 Scheduler；
//   - 不包含后端专有头文件或条件分支。
//
// 依赖：
//   Runtime/common/plan.h、kv_view.h、io_binder.h、Graph、Backend、Kernel。

#ifndef MOBILENET_RUNTIME_EXECUTOR_EXECUTOR_H_
#define MOBILENET_RUNTIME_EXECUTOR_EXECUTOR_H_

#include <memory>

#include "Runtime/common/kv_view.h"
#include "Runtime/common/plan.h"
#include "Runtime/common/status.h"

namespace mobilenet {

class Backend;

namespace runtime_graph {
class Graph;
}  // namespace runtime_graph

namespace runtime {

class IoBinder;

// Executor 统一接口。
//
// 功能：
//   Engine 在 Load 阶段调用 Init，在每步调用 Execute，在 Destroy 阶段调用
//   Deinit。描述符与 Graph 生命周期由 Engine 保证。
//
// 边界：
//   - 只允许单推理线程调用；
//   - Execute 必须完成 H2D/计算/D2H 与 Synchronize 后才返回；
//   - 失败时 result 中的请求标记为失败，不抛异常。
class IExecutor {
 public:
  virtual ~IExecutor() = default;

  // 初始化执行器。
  //
  // 参数：
  //   backend：后端实例，非空；
  //   graph：已重建并绑定 OpEnv 的图，非空；
  //   io_binder：Engine 实现的 I/O 绑定接口，非空。
  // 返回：
  //   成功返回 kOk；缺少 I/O op 或算子返回 kUnsupportedOp。
  virtual Status Init(Backend* backend, runtime_graph::Graph* graph,
                      IoBinder* io_binder) = 0;

  // 执行一个 Step。
  //
  // 参数：
  //   plan：本步执行计划，不可变快照；
  //   kv_view：KV 只读视图；
  //   result：输出参数，按请求写入 token 与成功/失败状态。
  // 返回：
  //   成功返回 kOk；I/O 绑定失败返回 kInvalidArgument；设备/Kernel 失败返回
  //   kDeviceError/kKernelError。
  virtual Status Execute(const ExecutionPlan& plan,
                         const KvCacheView& kv_view,
                         StepResult* result) = 0;

  // 反初始化执行器。
  //
  // 返回：
  //   成功返回 kOk；幂等。
  virtual Status Deinit() = 0;
};

// 创建默认 Executor。
//
// 返回：
//   非空 Executor；由 Engine 持有。实现方为 ExecBackend。
std::unique_ptr<IExecutor> CreateDefaultExecutor();

}  // namespace runtime
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_EXECUTOR_EXECUTOR_H_
