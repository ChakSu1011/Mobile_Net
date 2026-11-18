// 模块：Runtime 单步计划与执行结果契约
//
// 职责：
//   定义 Scheduler 产出、Executor 消费的 ExecutionPlan，以及 Executor 回传、
//   Engine/Scheduler 消费的 StepResult/StepStats；这些对象是调度与执行之间
//   唯一的跨模块数据契约，全部为值类型，不持有设备资源。
//
// 边界：
//   - 不包含调度策略、KV 分配策略或设备地址；
//   - 不依赖 Graph/Backend/Kernel 的具体类型；
//   - 请求生命周期由 sequence.h 的 RequestStatus 表达。
//
// 依赖：
//   Runtime/common/status.h、Runtime/common/sequence.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_COMMON_PLAN_H_
#define MOBILENET_RUNTIME_COMMON_PLAN_H_

#include <cstdint>
#include <vector>

#include "Runtime/common/sequence.h"
#include "Runtime/common/status.h"

namespace mobilenet {
namespace runtime {

// 单请求在本步的执行阶段。
//
// 说明：
//   kPrefill 表示本步处理 prompt 的下一块（可能未完成，见 is_chunk）；
//   kDecode 表示本步为已进入解码阶段的请求生成 1 个 token。
enum class RequestPhase {
  kPrefill = 0,
  kDecode = 1,
};

// 单请求在本步的执行片段。
//
// 功能：
//   描述某请求本步要计算的 token、位置、KV 写入位置与块表，供 Executor
//   构造输入缓冲和 Attention 元数据；同时给出回读 sample 输出的下标。
//
// 边界：
//   - 不携带 Request 对象或调度状态，Executor 只按本结构执行；
//   - block_table 与 slots 是调度时的快照，执行期间不得修改；
//   - draft_token_ids 为阶段三投机解码预留，阶段一/二必须为空。
struct PlanEntry {
  // 请求 ID，必须与 Scheduler 中的请求一致。
  uint64_t request_id = 0;

  // 本步阶段：Prefill 或 Decode。
  RequestPhase phase = RequestPhase::kPrefill;

  // 本片段的 token 在 packed token 缓冲中的起始下标。
  int32_t packed_offset = 0;

  // 本步 token 数：Prefill 为 chunk 长度，Decode 固定为 1。
  int32_t token_count = 0;

  // 本步要计算的 token id，长度为 token_count。
  std::vector<int64_t> token_ids;

  // 每个 token 的绝对位置（从 0 开始），长度为 token_count。
  std::vector<int64_t> positions;

  // 每个 token 的 KV 写入 slot，长度为 token_count。
  std::vector<int64_t> slots;

  // 本请求的逻辑块到物理块映射（只读快照）。
  std::vector<int32_t> block_table;

  // 本步之前该请求已处理的 token 数（prompt + generated）。
  int32_t context_length = 0;

  // 从 sample.out 回读 token 的 packed 下标；不产生 token 时为 -1。
  int32_t output_slot = -1;

  // Prefill 是否未完成：true 表示本块不是 prompt 的最后一块。
  bool is_chunk = false;

  // 阶段三投机解码的草稿 token；阶段一/二必须为空。
  std::vector<int64_t> draft_token_ids;
};

// 单步执行计划。
//
// 功能：
//   汇总本步所有请求的执行片段，是 Scheduler 与 Executor 之间唯一的
//   不可变快照。
//
// 边界：
//   - total_tokens 必须等于所有 entries 的 token_count 之和；
//   - total_tokens 不得超过 RuntimeConfig.max_num_tokens；
//   - step_id 单调递增，用于统计与阶段三异步释放栅栏。
struct ExecutionPlan {
  // 步序号，从 1 开始单调递增。
  uint64_t step_id = 0;

  // 本步总 token 数（所有请求之和）。
  int32_t total_tokens = 0;

  // 本步各请求的执行片段。
  std::vector<PlanEntry> entries;
};

// 单请求本步执行结果。
//
// 功能：
//   Executor 回读 sample 输出后，按请求填写 token 与成功/失败状态。
//
// 边界：
//   - 仅当 ok 为 true 时 token 有效；
//   - 失败时 error 必须为非 kOk 错误码，由 Engine 转交 Scheduler.Fail。
struct RequestStepResult {
  // 请求 ID。
  uint64_t request_id = 0;

  // 本步生成的 token（阶段一/二为 argmax 选中的 Token）。
  Token token;

  // 本请求本步是否执行成功。
  bool ok = true;

  // 失败原因；成功时为 kOk。
  ErrorCode error = ErrorCode::kOk;
};

// 单步执行结果集合。
//
// 功能：
//   携带本步所有请求的结果，由 Executor 返回给 Engine。
//
// 边界：
//   - 不包含调度状态或停止判定；
//   - 顺序不要求与 ExecutionPlan 一致，消费方按 request_id 匹配。
struct StepResult {
  // 各请求的执行结果。
  std::vector<RequestStepResult> requests;
};

// 单步统计。
//
// 功能：
//   记录本步的 token 数、队列长度、耗时与 KV 使用，供日志、性能基线与
//   回归脚本消费。
//
// 边界：
//   - 只读快照，不参与调度决策；
//   - 字段允许为 0 表示本步未采集。
struct StepStats {
  // 本步序号，与 ExecutionPlan.step_id 对应。
  uint64_t step_id = 0;

  // 本步总 token 数。
  int32_t total_tokens = 0;

  // 本步 Prefill token 数。
  int32_t prefill_tokens = 0;

  // 本步 Decode token 数。
  int32_t decode_tokens = 0;

  // 本步执行时的 running 请求数。
  int32_t running_requests = 0;

  // 本步执行时的 waiting 请求数。
  int32_t waiting_requests = 0;

  // Schedule 阶段耗时（纳秒）。
  int64_t schedule_ns = 0;

  // Execute 阶段耗时（纳秒）。
  int64_t execute_ns = 0;

  // 本步使用的 KV 块数。
  int64_t kv_used_blocks = 0;

  // 本步结束时的空闲 KV 块数。
  int64_t kv_free_blocks = 0;

  // 本步发生抢占的请求数。
  int32_t preempted_requests = 0;
};

}  // namespace runtime
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_COMMON_PLAN_H_
