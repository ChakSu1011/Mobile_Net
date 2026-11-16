// 模块：Runtime Scheduler 统一接口
//
// 职责：
//   定义 Engine 调用的 IScheduler：维护 Waiting/Running 队列、按
//   max_num_tokens 预算组批、执行 chunked prefill、KV 准入、停止判定与抢占，
//   产出 ExecutionPlan；并通过 Update 回写生成 token 与状态。
//
// 边界：
//   - 不调用设备/Kernel，不感知 KV 物理布局；
//   - 不实现停止 token 的文本逻辑（只按 token id 判定）；
//   - 不拥有请求之外的资源；KV 通过 KvAllocator 接口申请。
//
// 依赖：
//   Runtime/common/request.h、plan.h、model_config.h、kv_allocator.h。

#ifndef MOBILENET_RUNTIME_SCHEDULER_SCHEDULER_H_
#define MOBILENET_RUNTIME_SCHEDULER_SCHEDULER_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "Runtime/common/kv_allocator.h"
#include "Runtime/common/model_config.h"
#include "Runtime/common/plan.h"
#include "Runtime/common/request.h"
#include "Runtime/common/status.h"

namespace mobilenet {
namespace runtime {

// 调度配置。
//
// 功能：
//   由 Engine 从 manifest/ RuntimeConfig 合成；max_num_tokens 是单步硬预算。
//
// 边界：
//   - max_num_tokens 必须大于 0 且不超过图 S 维；
//   - max_prefill_chunk 为 0 时取 max_num_tokens；
//   - context_length 只用于停止判定与单请求上限推导。
struct SchedulerConfig {
  // 单步 token 预算（来自 manifest inference_config）。
  int32_t max_num_tokens = 0;

  // 并发请求上限。
  int32_t max_concurrent_requests = 0;

  // 单个 Prefill chunk 上限；0 表示等于 max_num_tokens。
  int32_t max_prefill_chunk = 0;

  // 模型上下文长度上限。
  int64_t context_length = 0;

  // 模型默认停止 token id。
  std::vector<int64_t> stop_ids;
};

// 调度统计。
//
// 功能：
//   供 EngineStats、日志与测试断言。
struct SchedulerStats {
  // 等待队列长度。
  int32_t waiting_requests = 0;

  // 运行队列长度。
  int32_t running_requests = 0;

  // 累计/当前被抢占请求数（由实现定义口径并注释）。
  int32_t preempted_requests = 0;

  // 空闲 KV 块数。
  int64_t free_blocks = 0;

  // KV 总块数。
  int64_t total_blocks = 0;
};

// Update 的结果。
//
// 功能：
//   Engine 据此判断请求是否结束、是否需要释放 KV 与触发回调。
struct UpdateResult {
  // 请求 ID。
  uint64_t request_id = 0;

  // 更新后的请求状态。
  RequestState state = RequestState::kRunning;

  // 是否刚刚进入终态（Finished/Cancelled/Failed）。
  bool finished = false;
};

// Scheduler 统一接口。
//
// 功能：
//   Engine 通过本接口完成请求入队、单步调度、结果回写、取消与状态查询。
//
// 边界：
//   - Schedule/Update/Fail/Stats 由推理线程调用；Merge/Stop/HasRequests
//     可能由调用方线程与推理线程并发调用，实现必须在内部加锁；
//   - Schedule 失败时不得留下半分配 KV；
//   - Update 只处理输出 token 与状态，不推进 num_computed_tokens。
class IScheduler {
 public:
  virtual ~IScheduler() = default;

  // 初始化调度器。
  //
  // 参数：
  //   config：调度预算与停止配置；
  //   kv_allocator：KV 分配接口，非空，生命周期由 Engine 保证；
  //   model：模型配置，用于上下文与停止 token 推导。
  // 返回：
  //   成功返回 kOk；配置非法返回 kInvalidArgument。
  virtual Status Init(const SchedulerConfig& config,
                      KvAllocator* kv_allocator,
                      const ModelConfig& model) = 0;

  // 加入一个请求。
  //
  // 参数：
  //   request：请求；id 为 0 时由调度器分配；
  //   request_id：输出参数，成功时写入最终请求 ID。
  // 返回：
  //   成功返回 kOk；prompt 为空、长度超限或无法容纳返回 kInvalidArgument/
  //   kDeviceOom。
  virtual Status Merge(const Request& request, uint64_t* request_id) = 0;

  // 生成本步执行计划。
  //
  // 参数：
  //   plan：输出参数；无请求可调度时 entries 为空、total_tokens 为 0。
  // 返回：
  //   成功返回 kOk；KV 分配失败且无法抢占返回 kDeviceOom；内部状态非法
  //   返回 kInternal。
  virtual Status Schedule(ExecutionPlan* plan) = 0;

  // 回写本步执行结果。
  //
  // 参数：
  //   request_id：请求 ID；
  //   tokens：本步新产生的 token（阶段一/二长度为 1；阶段三可多个）；
  //   executor_status：执行器报告的状态（默认 kRunning；失败时 kFailed）；
  //   result：输出参数，返回更新后的状态与是否结束。
  // 返回：
  //   成功返回 kOk；请求不存在返回 kRequestNotFound；已结束请求幂等忽略。
  // 说明：
  //   追加输出 token 并判停；不推进 num_computed_tokens（后者在 Schedule
  //   时按已调度 token 推进）。请求进入终态时，实现必须通过 KvAllocator::
  //   Release 释放其 KV；Engine 不重复释放。
  virtual Status Update(uint64_t request_id,
                        const std::vector<int64_t>& tokens,
                        RequestState executor_status,
                        UpdateResult* result) = 0;

  // 取消请求并释放其 KV。
  //
  // 参数：
  //   request_id：请求 ID。
  // 返回：
  //   成功返回 kOk；请求不存在返回 kRequestNotFound。
  virtual Status Stop(uint64_t request_id) = 0;

  // 标记请求失败并释放其 KV。
  //
  // 参数：
  //   request_id：请求 ID；
  //   code：失败原因，不得为 kOk。
  // 返回：
  //   成功返回 kOk；请求不存在返回 kRequestNotFound。
  virtual Status Fail(uint64_t request_id, ErrorCode code) = 0;

  // 返回全部请求的状态快照。
  virtual std::vector<RequestSnapshot> Snapshot() const = 0;

  // 返回调度统计。
  virtual SchedulerStats Stats() const = 0;

  // 是否存在未结束请求。
  virtual bool HasRequests() const = 0;
};

// 创建默认 Scheduler。
//
// 返回：
//   非空 Scheduler；由 Engine 持有。实现方为 RuntimeCore。
std::unique_ptr<IScheduler> CreateDefaultScheduler();

}  // namespace runtime
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_SCHEDULER_SCHEDULER_H_
