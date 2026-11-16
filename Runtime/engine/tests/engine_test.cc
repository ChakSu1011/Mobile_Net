// 模块：Runtime Engine 生命周期单元测试
//
// 职责：
//   验证 Engine 在无模块/未加载/已销毁状态下的状态机、参数校验、错误码与
//   幂等销毁；默认模块工厂使用返回 nullptr 的测试桩，用于覆盖"默认实现
//   不可用"的失败路径。
//
// 边界：
//   - 不验证调度、KV、执行、后端与真实模型包；
//   - 不验证 threaded 推理循环与 Load 成功路径（需要 Fake 模块，后续补充）；
//   - 只用于 Engine 模块的早期回归。
//
// 依赖：
//   Runtime/engine/engine.cc、Runtime/common/resource_stack.cc、Graph 静态库。

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

#include "Graph/include/storage.h"
#include "Runtime/destroy/destroyer.h"
#include "Runtime/engine/engine.h"
#include "Runtime/engine/engine_modules.h"
#include "Runtime/executor/executor.h"
#include "Runtime/kvcache/kv_cache_manager.h"
#include "Runtime/loader/loader.h"
#include "Runtime/scheduler/scheduler.h"

namespace mobilenet {
namespace runtime {

// 以下默认工厂桩返回空指针，用于验证 Engine 在默认模块缺失时的错误路径。
// 真实实现由各模块在后续阶段提供，届时删除本文件中的桩定义。
std::unique_ptr<ILoader> CreateDefaultLoader() { return nullptr; }
std::unique_ptr<IScheduler> CreateDefaultScheduler() { return nullptr; }
std::unique_ptr<IExecutor> CreateDefaultExecutor() { return nullptr; }
std::unique_ptr<IKVCache> CreateDefaultKVCache() { return nullptr; }
std::unique_ptr<IDestroyer> CreateDefaultDestroyer() { return nullptr; }

}  // namespace runtime
}  // namespace mobilenet

namespace {

// 断言计数。
int g_failures = 0;

// 检查布尔条件。
//
// 参数：
//   condition：期望为真的条件；
//   message：失败时输出的用例描述。
void Check(bool condition, const char* message) {
  if (!condition) {
    std::printf("FAIL: %s\n", message);
    ++g_failures;
  }
}

// 检查 Status 的错误码。
//
// 参数：
//   status：实际状态；
//   expected：期望错误码；
//   message：失败时输出的用例描述。
void CheckStatus(const mobilenet::Status& status,
                 mobilenet::ErrorCode expected, const char* message) {
  Check(status.code() == expected, message);
}

}  // namespace

// Engine 生命周期测试入口。
//
// 返回：
//   全部用例通过返回 0；否则返回 1。
int main() {
  using mobilenet::ErrorCode;
  using mobilenet::runtime::Engine;
  using mobilenet::runtime::EngineConfig;
  using mobilenet::runtime::EngineModules;
  using mobilenet::runtime::EngineState;
  using mobilenet::runtime::EngineStats;
  using mobilenet::runtime::Request;

  {
    Engine engine;
    Check(engine.state() == EngineState::kCreated, "initial state is CREATED");
    const EngineStats stats = engine.stats();
    Check(stats.total_steps == 0 && stats.total_requests == 0 &&
              stats.total_tokens == 0,
          "initial stats are zero");
    CheckStatus(engine.SetModules(EngineModules{}), ErrorCode::kOk,
                "SetModules accepts empty modules");

    EngineConfig empty_config;
    CheckStatus(engine.Load(empty_config), ErrorCode::kInvalidArgument,
                "Load rejects empty package_dir");
    CheckStatus(engine.RunOnce(nullptr), ErrorCode::kStateError,
                "RunOnce rejects CREATED state");
    Request request;
    request.prompt_ids = {1, 2, 3};
    uint64_t request_id = 0;
    CheckStatus(engine.AddRequest(request, &request_id),
                ErrorCode::kStateError,
                "AddRequest rejects CREATED state");
    CheckStatus(engine.CancelRequest(1), ErrorCode::kStateError,
                "CancelRequest rejects CREATED state");
    CheckStatus(engine.Update(1, {1}), ErrorCode::kStateError,
                "Update rejects CREATED state");
    CheckStatus(engine.RunUntilIdle(1), ErrorCode::kStateError,
                "RunUntilIdle rejects CREATED state");
    mobilenet::runtime_graph::Storage storage;
    CheckStatus(engine.SetInput("input.input_ids", storage),
                ErrorCode::kStateError,
                "SetInput rejects CREATED state");
    CheckStatus(engine.SetOutput("output.tokens", storage),
                ErrorCode::kStateError,
                "SetOutput rejects CREATED state");
    CheckStatus(engine.Destroy(), ErrorCode::kOk,
                "Destroy succeeds from CREATED");
    Check(engine.state() == EngineState::kDestroyed,
          "state becomes DESTROYED");
    CheckStatus(engine.Destroy(), ErrorCode::kOk, "Destroy is idempotent");
    CheckStatus(engine.Load(empty_config), ErrorCode::kStateError,
                "Load rejects DESTROYED state");
  }

  {
    Engine engine;
    EngineConfig config;
    config.package_dir = "missing-package";
    CheckStatus(engine.Load(config), ErrorCode::kUnsupportedOp,
                "Load fails when default modules are unavailable");
    Check(engine.state() == EngineState::kCreated,
          "failed load returns to CREATED");
    CheckStatus(engine.Destroy(), ErrorCode::kOk,
                "Destroy succeeds after failed load");
  }

  if (g_failures != 0) {
    std::printf("%d engine checks failed\n", g_failures);
    return 1;
  }
  std::printf("all engine lifecycle checks passed\n");
  return 0;
}
