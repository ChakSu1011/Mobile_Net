// 模块：Runtime Destroy 统一接口
//
// 职责：
//   定义 Engine 调用的 IDestroyer：按依赖逆序停止调度、同步设备、释放
//   OpEnv/Kernel、缓冲、GraphEnv 与 Backend，并保证幂等；加载失败回滚复用
//   同一路径。
//
// 边界：
//   - 不创建资源，只释放；
//   - 不直接调用设备释放 API，全部经 Backend/OpEnv/ResourceStack；
//   - 不负责删除 Engine 自身（由 Engine 析构完成）。
//
// 依赖：
//   Runtime/common/status.h、Runtime/common/resource_stack.h。

#ifndef MOBILENET_RUNTIME_DESTROY_DESTROYER_H_
#define MOBILENET_RUNTIME_DESTROY_DESTROYER_H_

#include <memory>

#include "Runtime/common/resource_stack.h"
#include "Runtime/common/status.h"

namespace mobilenet {

class Backend;

namespace runtime_graph {
class Graph;
}  // namespace runtime_graph

namespace runtime {

class IScheduler;
class IKVCache;
class IExecutor;

// 销毁上下文。
//
// 功能：
//   携带 Destroy 需要访问的模块指针；全部为非拥有裸指针，生命周期由
//   Engine 保证。
//
// 边界：
//   - backend_initialized 为 false 时跳过 Synchronize/Deinit；
//   - resource_stack 为空时跳过缓冲释放；
//   - 各指针允许为空（部分加载失败场景）。
struct DestroyContext {
  // 调度器；用于停止接收新请求。
  IScheduler* scheduler = nullptr;

  // KV 管理器；用于释放块与缓冲。
  IKVCache* kv_cache = nullptr;

  // 执行器；用于反初始化。
  IExecutor* executor = nullptr;

  // 图；用于 ClearOpEnv/ClearGraphEnv。
  runtime_graph::Graph* graph = nullptr;

  // 后端；用于 Synchronize/Deinit。
  Backend* backend = nullptr;

  // 资源释放栈。
  ResourceStack* resource_stack = nullptr;

  // 后端是否已成功初始化。
  bool backend_initialized = false;
};

// Destroy 统一接口。
//
// 功能：
//   Engine 在正常销毁与加载失败回滚时调用；实现按固定阶段逆序释放。
//
// 边界：
//   - 幂等：重复调用安全；
//   - 单个释放失败记录并继续，最终返回首个错误；
//   - 不阻塞推理线程；调用方（Engine）负责先停线程。
class IDestroyer {
 public:
  virtual ~IDestroyer() = default;

  // 执行销毁。
  //
  // 参数：
  //   context：待释放模块与资源栈。
  // 返回：
  //   全部成功返回 kOk；否则返回首个失败状态。
  virtual Status Destroy(const DestroyContext& context) = 0;
};

// 创建默认 Destroyer。
//
// 返回：
//   非空 Destroyer；由 Engine 持有。实现方为 RuntimeCore。
std::unique_ptr<IDestroyer> CreateDefaultDestroyer();

}  // namespace runtime
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_DESTROY_DESTROYER_H_
