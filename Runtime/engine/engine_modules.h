// 模块：Runtime Engine 可注入模块集合
//
// 职责：
//   定义 Engine 的组合根配置 EngineModules：允许 Service/测试在 Load 前替换
//   Loader/Scheduler/Executor/Destroyer 实现；Loader 与 Destroyer 成对注入，
//   对应资源的 Init/UnInit 生命周期。
//
// 边界：
//   - 只在 Engine CREATED 状态使用；
//   - Graph/Backend 由 Loader 产出，不在本结构中；
//   - KVCache 由 Engine 内部创建，不在此注入；
//   - 本头文件会包含各模块接口，因此不被 engine.h 直接包含，避免把
//     Backend/Loader 依赖带给所有 Engine 使用者。
//
// 依赖：
//   Runtime/loader/loader.h、Runtime/scheduler/scheduler.h、
//   Runtime/executor/executor.h、Runtime/destroy/destroyer.h。

#ifndef MOBILENET_RUNTIME_ENGINE_ENGINE_MODULES_H_
#define MOBILENET_RUNTIME_ENGINE_ENGINE_MODULES_H_

#include <memory>

#include "Runtime/destroy/destroyer.h"
#include "Runtime/executor/executor.h"
#include "Runtime/loader/loader.h"
#include "Runtime/scheduler/scheduler.h"

namespace mobilenet {
namespace runtime {

// Engine 可注入模块集合。
//
// 功能：
//   Service/测试在 `Engine::SetModules` 中传入；为空指针的字段由 Engine
//   创建默认实现。
//
// 边界：
//   - Loader 与 Destroyer 应成对替换：Loader 创建的资源由 Destroyer 按
//     ResourceStack 逆序释放；
//   - Graph/Backend 由 Loader 在 Load 时产出，不能预先注入；
//   - 注入后所有权移交 Engine。
struct EngineModules {
  // 加载器实现；为空时使用默认实现。
  std::unique_ptr<ILoader> loader;

  // 调度器实现；为空时使用默认实现。
  std::unique_ptr<IScheduler> scheduler;

  // 执行器实现；为空时使用默认实现（ExecBackend）。
  std::unique_ptr<IExecutor> executor;

  // 销毁器实现；为空时使用默认实现；建议与 loader 成对替换。
  std::unique_ptr<IDestroyer> destroyer;
};

}  // namespace runtime
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_ENGINE_ENGINE_MODULES_H_
