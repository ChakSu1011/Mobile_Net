// 模块：Runtime Engine 总接口
//
// 职责：
//   作为 Runtime 的唯一门面与总接口：持有 Loader/Graph/Backend/Scheduler/
//   Executor/KVCache/Destroy 成员指针，编排加载、单步推理、请求生命周期与
//   资源释放；对外提供 token 级 API 供阶段二 Service 层调用。
//
// 边界：
//   - 不实现调度、KV、执行、后端算法，只通过统一接口编排；
//   - 不接触文本/Tokenizer；
//   - 不直接调用 OpEnv/Kernel；执行链经 IExecutor 进入。
//
// 依赖：
//   Runtime/common/*、Runtime/loader/loader.h、Runtime/scheduler/scheduler.h、
//   Runtime/kvcache/kv_cache_manager.h、Runtime/executor/executor.h、
//   Runtime/destroy/destroyer.h、Graph/include/storage.h。

#ifndef MOBILENET_RUNTIME_ENGINE_ENGINE_H_
#define MOBILENET_RUNTIME_ENGINE_ENGINE_H_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "Graph/include/storage.h"
#include "Runtime/common/io_binder.h"
#include "Runtime/common/model_config.h"
#include "Runtime/common/op_env_factory.h"
#include "Runtime/common/plan.h"
#include "Runtime/common/request.h"
#include "Runtime/common/resource_stack.h"
#include "Runtime/common/runtime_config.h"
#include "Runtime/common/status.h"
#include "Runtime/destroy/destroyer.h"
#include "Runtime/executor/executor.h"
#include "Runtime/kvcache/kv_cache_manager.h"
#include "Runtime/loader/loader.h"
#include "Runtime/scheduler/scheduler.h"

namespace mobilenet {

class Backend;

namespace runtime_graph {
class Graph;
}  // namespace runtime_graph

namespace runtime {

// Engine 生命周期状态。
//
// 迁移：
//   CREATED → Load → LOADED → Start → RUNNING → Destroy → DESTROYING →
//   DESTROYED；加载失败回到 CREATED；DESTROYED 为终态。
enum class EngineState {
  kCreated = 0,
  kLoaded = 1,
  kRunning = 2,
  kDestroying = 3,
  kDestroyed = 4,
};

// Engine 配置。
//
// 功能：
//   Service 层指定模型包与运行参数；Engine 在 Load 时转换为各模块配置。
struct EngineConfig {
  // 模型包目录。
  std::string package_dir;

  // 运行参数。
  RuntimeConfig runtime;
};

// 可注入的模块实现。
//
// 功能：
//   Service/测试可在 Load 前替换部分模块（例如不同版本的 Scheduler）；
//   未注入的模块由 Engine 创建默认实现。
//
// 边界：
//   - 只在 Engine CREATED 状态注入；
//   - Graph/Backend 由 Loader 产出，不在本结构中；
//   - 注入后所有权移交 Engine。
struct EngineModules {
  // 加载器实现；为空时使用默认实现。
  std::unique_ptr<ILoader> loader;

  // 调度器实现；为空时使用默认实现。
  std::unique_ptr<IScheduler> scheduler;

  // 执行器实现；为空时使用默认实现（ExecBackend）。
  std::unique_ptr<IExecutor> executor;

  // KVCache 实现；为空时使用默认实现。
  std::unique_ptr<IKVCache> kv_cache;

  // Destroy 实现；为空时使用默认实现。
  std::unique_ptr<IDestroyer> destroyer;
};

// Engine 累计统计。
//
// 功能：
//   供 Service 查询与日志；字段由实现按步更新。
struct EngineStats {
  // 累计执行步数。
  uint64_t total_steps = 0;

  // 累计接收请求数。
  uint64_t total_requests = 0;

  // 累计生成 token 数。
  uint64_t total_tokens = 0;

  // 当前运行请求数。
  int32_t running_requests = 0;

  // 当前等待请求数。
  int32_t waiting_requests = 0;

  // 当前已用 KV 块数。
  int64_t kv_used_blocks = 0;

  // 当前空闲 KV 块数。
  int64_t kv_free_blocks = 0;

  // 最近一步耗时（纳秒）。
  int64_t last_step_ns = 0;
};

// 流式 token 回调。
//
// 参数：
//   request_id：请求 ID；
//   token：新生成的 token；
//   finished：该请求是否已结束。
// 说明：
//   在推理线程同步调用；实现不得阻塞、不得重入 Engine。
using TokenCallback =
    std::function<void(uint64_t request_id, int64_t token, bool finished)>;

// Engine 总接口。
//
// 功能：
//   Service 层通过本类完成 Load/Start/AddRequest/CancelRequest/Update/
//   RunOnce/RunUntilIdle/Destroy 与 I/O 绑定。
//
// 边界：
//   - 单线程模式由调用方驱动 RunOnce；threaded 模式由内部线程驱动；
//   - Load/Start/Destroy/SetModules 不得并发调用；
//   - Engine 不实现任何模块算法。
class Engine final : public IoBinder {
 public:
  // 构造 Engine；模块指针为空，等待 SetModules 或默认创建。
  Engine();

  // 析构 Engine；若未 Destroy 则执行受控释放。
  ~Engine() override;

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  Engine(Engine&&) = delete;
  Engine& operator=(Engine&&) = delete;

  // 注入模块实现。
  //
  // 参数：
  //   modules：模块实现；空指针项使用默认实现。
  // 返回：
  //   成功返回 kOk；非 CREATED 状态返回 kStateError。
  Status SetModules(EngineModules modules);

  // 加载模型包并创建运行时资源。
  //
  // 参数：
  //   config：包目录与运行参数。
  // 返回：
  //   成功返回 kOk（state=LOADED）；失败返回对应错误码并回滚到 CREATED。
  Status Load(const EngineConfig& config);

  // 启动推理。
  //
  // 返回：
  //   成功返回 kOk（state=RUNNING）；threaded 模式启动推理线程。
  Status Start();

  // 加入请求。
  //
  // 参数：
  //   request：请求；id 为 0 时由 Engine 分配；
  //   request_id：输出参数，成功时写入请求 ID。
  // 返回：
  //   成功返回 kOk；未 LOADED/RUNNING、参数非法或容量不足返回对应错误码。
  // 说明：
  //   线程安全，可由任意调用方线程调用。
  Status AddRequest(const Request& request, uint64_t* request_id);

  // 取消请求。
  //
  // 参数：
  //   request_id：请求 ID。
  // 返回：
  //   成功返回 kOk；请求不存在返回 kRequestNotFound。
  // 说明：
  //   线程安全；执行中的请求由调度器在安全点取消。
  Status CancelRequest(uint64_t request_id);

  // 回写生成 token 与状态。
  //
  // 参数：
  //   request_id：请求 ID；
  //   tokens：本步新产生的 token（阶段一/二为 1 个）；
  //   status：执行器报告的状态，默认 kRunning。
  // 返回：
  //   成功返回 kOk；请求不存在返回 kRequestNotFound。
  // 说明：
  //   正常路径由 RunOnce 内部调用；外部仅在自定义执行路径下使用。
  Status Update(uint64_t request_id, const std::vector<int64_t>& tokens,
                RequestState status = RequestState::kRunning);

  // 执行一个 Step。
  //
  // 参数：
  //   stats：输出参数，写入本步统计；允许为空。
  // 返回：
  //   成功返回 kOk；threaded 模式返回 kStateError；设备/执行错误返回对应
  //   错误码。
  // 说明：
  //   只允许一个调用者线程；单线程模式使用。
  Status RunOnce(StepStats* stats);

  // 驱动到空闲。
  //
  // 参数：
  //   timeout_ms：超时毫秒；0 表示不等待。
  // 返回：
  //   成功返回 kOk；超时返回 kTimeout。
  Status RunUntilIdle(uint32_t timeout_ms);

  // 销毁全部运行时资源。
  //
  // 返回：
  //   成功返回 kOk（state=DESTROYED）；释放失败返回首个错误，仍进入
  //   DESTROYED。
  // 说明：
  //   幂等；会停止并 join 推理线程。
  Status Destroy();

  // 设置流式 token 回调。
  //
  // 参数：
  //   callback：回调；传空表示清除。
  void SetTokenCallback(TokenCallback callback);

  // 绑定图输入主机缓冲（IoBinder 实现）。
  //
  // 参数：
  //   name：Input op 名称；
  //   host_storage：主机 Storage 描述。
  // 返回：
  //   成功返回 kOk；名称/尺寸/精度不匹配返回 kInvalidArgument。
  Status SetInput(const std::string& name,
                  const runtime_graph::Storage& host_storage) override;

  // 绑定图输出主机缓冲（IoBinder 实现）。
  //
  // 参数：
  //   name：Output op 名称；
  //   host_storage：主机 Storage 描述。
  // 返回：
  //   成功返回 kOk；名称/尺寸不匹配返回 kInvalidArgument。
  Status SetOutput(const std::string& name,
                   const runtime_graph::Storage& host_storage) override;

  // 返回当前状态。
  EngineState state() const;

  // 返回累计统计。
  EngineStats stats() const;

 private:
  // 运行时状态。
  EngineState state_ = EngineState::kCreated;

  // 运行配置与模型配置。
  RuntimeConfig runtime_config_;
  ModelConfig model_config_;

  // 模块成员：全部以指针形式持有，便于替换实现。
  std::unique_ptr<ILoader> loader_;
  std::unique_ptr<runtime_graph::Graph> graph_;
  std::unique_ptr<Backend> backend_;
  std::unique_ptr<IScheduler> scheduler_;
  std::unique_ptr<IExecutor> executor_;
  std::unique_ptr<IKVCache> kv_cache_;
  std::unique_ptr<IDestroyer> destroyer_;

  // 内部辅助对象。
  std::unique_ptr<OpEnvFactory> op_env_factory_;
  std::unique_ptr<ResourceStack> resource_stack_;

  // 回调与统计。
  TokenCallback token_callback_;
  EngineStats stats_;

  // 线程与同步状态。
  std::thread inference_thread_;
  mutable std::mutex mutex_;
  std::condition_variable idle_cv_;
  std::atomic<bool> stop_requested_{false};

  // 设备级错误标记。
  bool fatal_ = false;
  ErrorCode fatal_code_ = ErrorCode::kOk;
};

}  // namespace runtime
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_ENGINE_ENGINE_H_
