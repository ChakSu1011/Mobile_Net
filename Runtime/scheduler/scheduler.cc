// 模块：Runtime Scheduler 默认实现
//
// 职责：
//   实现 IScheduler：请求合并、连续批处理、chunked prefill、完整历史重算、
//   停止判定、取消/失败终态与调度统计；维护每个请求的 Sequence 与状态。
//
// 边界：
//   - 不调用设备/Kernel，不感知 KV 物理布局；
//   - recompute_attention=true 时只生成“完整历史重算”计划，不申请 KV，
//     slots/block_table 为空；
//   - recompute_attention=false 时通过 KvAllocator 申请/释放块并组装
//     block_table/slots；
//   - 不实现抢占：KV 不足时保留请求等待下一步，preempted_requests 恒为 0；
//   - 不实现阶段三投机解码与 Beam，tokens 向量按顺序追加，遇停止 token
//     立即截断。
//
// 依赖：
//   Runtime/common/{sequence,plan,kv_allocator,model_config,status}.h。

#include "Runtime/scheduler/scheduler.h"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "Utils/cpp/types.h"

namespace mobilenet {
namespace runtime {
namespace {

// 判断请求状态是否为终态。
//
// 参数：
//   status：请求状态。
// 返回：
//   kStopped/kIllegal 返回 true；其余返回 false。
// 说明：
//   终态请求不再进入调度，但保留在请求表中以支持幂等 Repeat。
Bool IsTerminalStatus(RequestStatus status) {
  return status == RequestStatus::kStopped ||
         status == RequestStatus::kIllegal;
}

// 判断请求是否仍占用调度队列。
//
// 参数：
//   status：请求状态。
// 返回：
//   kWaiting/kChunkedPrefill/kRunning 返回 true。
Bool IsActiveStatus(RequestStatus status) {
  return status == RequestStatus::kWaiting ||
         status == RequestStatus::kChunkedPrefill ||
         status == RequestStatus::kRunning;
}

// 判断 token id 列表是否包含指定 token。
//
// 参数：
//   ids：停止 token 列表；
//   token_id：待检查 token。
// 返回：
//   包含返回 true。
Bool ContainsToken(const std::vector<I64>& ids, I64 token_id) {
  return std::find(ids.begin(), ids.end(), token_id) != ids.end();
}

// 把 I64 安全裁剪到 I32 范围。
//
// 参数：
//   value：待转换值。
// 返回：
//   [0, INT32_MAX] 内的值；负数返回 0。
I32 ClampToI32(I64 value) {
  if (value <= 0) {
    return 0;
  }
  if (value > static_cast<I64>(std::numeric_limits<I32>::max())) {
    return std::numeric_limits<I32>::max();
  }
  return static_cast<I32>(value);
}

// 调度器内部请求状态。
//
// 功能：
//   在 Request 之外保存 Sequence、请求状态、已计算 token 数与 KV 占用标记；
//   仅在 SchedulerImpl 的锁内读写。
//
// 边界：
//   - sequence.history_token_ids 同时承载 prompt 与已生成 token；
//   - computed_tokens 表示已完成 KV 写入（或重算模式已调度）的 token 数，
//     由 Schedule 推进，Update 不修改。
struct RequestState {
  // 请求 ID。
  U64 request_id = 0;

  // 序列状态（history/current_token/generated_count 等）。
  Sequence sequence;

  // 请求状态。
  RequestStatus status = RequestStatus::kWaiting;

  // 请求级采样参数副本。
  SampleParams sample_param;

  // 已计算（已写 KV）token 数。
  I32 computed_tokens = 0;

  // 是否已向 KvAllocator 申请过容量。
  Bool kv_reserved = false;
};

// 用请求状态填充 Response。
//
// 参数：
//   state：请求内部状态；
//   status：本步要回传的状态；
//   response：输出参数，非空。
// 返回：
//   无；直接覆盖 response 的 request_id/status/sequences/sample_param。
void FillResponse(const RequestState& state, RequestStatus status,
                  Response* response) {
  response->request_id = state.request_id;
  response->status = status;
  response->sequences.assign(1, state.sequence);
  response->sample_param = state.sample_param;
}

// 默认调度器实现。
//
// 功能：
//   组合 Waiting/Running 请求、token 预算与 KvAllocator，产出 ExecutionPlan；
//   维护请求生命周期并生成 Response。
//
// 边界：
//   - 所有公共方法内部加锁；Merge/Stop/HasRequests 可与推理线程并发；
//   - Schedule 失败时不修改计划内容；KV 分配失败时返回的计划为空；
//   - 已结束请求保留在 requests_ 中，重复 Stop/Update/Fail 幂等返回。
class SchedulerImpl final : public IScheduler {
 public:
  SchedulerImpl() = default;

  ~SchedulerImpl() override = default;

  // 初始化调度器。
  //
  // 参数与返回见 IScheduler::Init；额外校验 max_concurrent_requests 与
  // vocab_size 范围，非法时返回 kInvalidArgument。
  Status Init(const SchedulerConfig& config, KvAllocator* kv_allocator,
              const ModelConfig& model) override;

  // 加入请求。
  //
  // 返回：
  //   kOk；prompt 为空/超限/含非法 token 返回 kInvalidArgument；重复
  //   request_id 返回 kInvalidArgument；并发已满返回 kDeviceOom。
  Status Merge(const Request& request, U64* request_id) override;

  // 生成单步计划。
  //
  // 返回：
  //   kOk；参数/状态非法返回 kInvalidArgument/kStateError；所有候选请求
  //   KV 分配失败且计划为空时返回首个 KV 错误（通常 kDeviceOom）。
  Status Schedule(ExecutionPlan* plan) override;

  // 回写本步 token 并生成 Response。
  //
  // 说明：
  //   chunked prefill 未完成（kChunkedPrefill）时忽略输出 token；终态请求
  //   幂等返回原状态；非终态返回 kRunning。
  Status Update(U64 request_id, const std::vector<Token>& tokens,
                Response* response) override;

  // 取消请求。
  //
  // 返回：
  //   kOk；请求不存在返回 kRequestNotFound；已结束请求幂等。
  Status Stop(U64 request_id, Response* response) override;

  // 标记请求失败。
  //
  // 返回：
  //   kOk；code 为 kOk 返回 kInvalidArgument；请求不存在返回
  //   kRequestNotFound；已结束请求幂等保留原状态。
  Status Fail(U64 request_id, ErrorCode code, Response* response) override;

  // 返回全部请求的状态快照。
  std::vector<RequestSnapshot> Snapshot() const override;

  // 返回调度统计。
  //
  // 说明：
  //   waiting_requests 含 kWaiting/kChunkedPrefill；running_requests 只含
  //   kRunning；preempted_requests 恒为 0（本阶段未实现抢占）。
  SchedulerStats Stats() const override;

  // 是否存在未结束请求。
  Bool HasRequests() const override;

 private:
  // 计算请求的有效最大生成 token 数。
  //
  // 参数：
  //   state：请求状态。
  // 返回：
  //   生成上限；由 sample.max_tokens 或 context_length-prompt 推导，并在
  //   重算模式下再受 max_num_tokens-prompt 约束。
  I64 EffectiveMaxTokens(const RequestState& state) const;

  // 判断请求是否应停止。
  //
  // 参数：
  //   state：已追加本步 token 的请求状态。
  // 返回：
  //   达到 max_tokens/context_length/重算预算，或生成数不小于 min_tokens
  //   且命中停止 token 时返回 true。
  Bool ShouldStop(const RequestState& state) const;

  // 释放请求已申请的 KV（若曾申请）。
  //
  // 参数：
  //   state：请求状态；Release 后把 kv_reserved 置 false。
  void ReleaseIfReserved(RequestState* state);

  // 统计未结束请求数。
  I32 ActiveRequestCount() const;

  // 生成完整历史重算计划。
  //
  // 参数：
  //   plan：输出计划，非空。
  // 返回：
  //   kOk；内部状态非法返回 kInternal。
  Status ScheduleRecompute(ExecutionPlan* plan);

  // 生成 KV 模式（chunked prefill + decode）计划。
  //
  // 参数：
  //   plan：输出计划，非空。
  // 返回：
  //   kOk；计划为空且 KV 分配失败时返回首个 KV 错误。
  Status ScheduleWithKv(ExecutionPlan* plan);

  // 互斥锁，保护全部请求状态与计数器。
  mutable std::mutex mutex_;

  // 是否已完成 Init。
  Bool initialized_ = false;

  // 单步 token 预算。
  I32 max_num_tokens_ = 0;

  // 单个 Prefill chunk 上限。
  I32 max_prefill_chunk_ = 0;

  // 并发未结束请求上限。
  I32 max_concurrent_requests_ = 0;

  // 模型上下文长度。
  I64 context_length_ = 0;

  // 词表大小；0 表示未知，不做 token 上下界校验。
  I32 vocab_size_ = 0;

  // 是否启用完整历史重算模式。
  Bool recompute_attention_ = false;

  // 模型默认停止 token。
  std::vector<I64> default_stop_ids_;

  // KV 分配接口；非拥有。
  KvAllocator* kv_allocator_ = nullptr;

  // 请求表；终态请求保留用于幂等查询。
  std::map<U64, RequestState> requests_;

  // 下一个自动分配的请求 ID。
  U64 next_request_id_ = 1;

  // 下一个计划步骤号，从 1 开始。
  U64 next_step_id_ = 1;
};

Status SchedulerImpl::Init(const SchedulerConfig& config,
                           KvAllocator* kv_allocator,
                           const ModelConfig& model) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (kv_allocator == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Scheduler::Init: kv_allocator must not be null");
  }
  if (config.max_num_tokens <= 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Scheduler::Init: max_num_tokens must be positive");
  }
  if (config.max_concurrent_requests < 0) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "Scheduler::Init: max_concurrent_requests must not be negative");
  }
  if (config.max_prefill_chunk < 0) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "Scheduler::Init: max_prefill_chunk must not be negative");
  }
  const I64 context_length =
      config.context_length > 0 ? config.context_length : model.context_length;
  if (context_length <= 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Scheduler::Init: context_length must be positive");
  }
  if (model.vocab_size < 0 ||
      model.vocab_size >
          static_cast<I64>(std::numeric_limits<I32>::max())) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Scheduler::Init: vocab_size is out of range");
  }

  max_num_tokens_ = config.max_num_tokens;
  max_prefill_chunk_ =
      config.max_prefill_chunk > 0
          ? std::min(config.max_prefill_chunk, config.max_num_tokens)
          : config.max_num_tokens;
  max_concurrent_requests_ =
      config.max_concurrent_requests > 0 ? config.max_concurrent_requests
                                         : config.max_num_tokens;
  context_length_ = context_length;
  vocab_size_ = static_cast<I32>(model.vocab_size);
  recompute_attention_ = config.recompute_attention;
  default_stop_ids_ = config.stop_ids.empty() ? model.stop_token_ids
                                              : config.stop_ids;
  kv_allocator_ = kv_allocator;
  requests_.clear();
  next_request_id_ = 1;
  next_step_id_ = 1;
  initialized_ = true;
  return Status::Ok();
}

Status SchedulerImpl::Merge(const Request& request, U64* request_id) {
  if (request_id == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Scheduler::Merge: request_id must not be null");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "Scheduler::Merge: scheduler is not initialized");
  }
  if (request.prompt_token_ids.empty()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Scheduler::Merge: prompt_token_ids must not be empty");
  }
  if (!request.multimodal_inputs.empty()) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "Scheduler::Merge: multimodal input is not supported");
  }
  if (request.sample_param.max_tokens < 0 ||
      request.sample_param.min_tokens < 0) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "Scheduler::Merge: max_tokens/min_tokens must not be negative");
  }
  const I64 prompt_size =
      static_cast<I64>(request.prompt_token_ids.size());
  if (prompt_size >
      static_cast<I64>(std::numeric_limits<I32>::max())) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Scheduler::Merge: prompt is too long");
  }
  if (prompt_size >= context_length_) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "Scheduler::Merge: prompt leaves no room in context_length");
  }
  if (recompute_attention_ && prompt_size >= max_num_tokens_) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "Scheduler::Merge: prompt leaves no room in recompute budget");
  }
  for (I64 token_id : request.prompt_token_ids) {
    if (token_id < 0 ||
        (vocab_size_ > 0 && token_id >= static_cast<I64>(vocab_size_))) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "Scheduler::Merge: prompt contains invalid token id");
    }
  }
  if (request.request_id != 0 &&
      requests_.find(request.request_id) != requests_.end()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Scheduler::Merge: duplicate request_id");
  }
  if (ActiveRequestCount() >= max_concurrent_requests_) {
    return MakeStatus(ErrorCode::kDeviceOom,
                      "Scheduler::Merge: max_concurrent_requests reached");
  }

  U64 assigned_id = request.request_id;
  if (assigned_id == 0) {
    const U64 kMaxRequestId = std::numeric_limits<U64>::max();
    while (requests_.find(next_request_id_) != requests_.end()) {
      if (next_request_id_ == 0 || next_request_id_ == kMaxRequestId) {
        return MakeStatus(ErrorCode::kDeviceOom,
                          "Scheduler::Merge: request_id space exhausted");
      }
      ++next_request_id_;
    }
    if (next_request_id_ == 0 || next_request_id_ == kMaxRequestId) {
      return MakeStatus(ErrorCode::kDeviceOom,
                        "Scheduler::Merge: request_id space exhausted");
    }
    assigned_id = next_request_id_++;
  } else if (assigned_id == std::numeric_limits<U64>::max()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Scheduler::Merge: request_id is out of range");
  } else if (assigned_id >= next_request_id_) {
    next_request_id_ = assigned_id + 1;
  }

  RequestState state;
  state.request_id = assigned_id;
  state.sequence.sequence_id = assigned_id;
  state.sequence.history_token_ids = request.prompt_token_ids;
  state.sequence.prompt_count = static_cast<I32>(prompt_size);
  state.sequence.generated_count = 0;
  state.sequence.vocab_size = vocab_size_;
  state.sequence.current_token = Token{};
  state.sample_param = request.sample_param;
  state.status = RequestStatus::kWaiting;
  state.computed_tokens = 0;
  state.kv_reserved = false;
  requests_.emplace(assigned_id, std::move(state));
  *request_id = assigned_id;
  return Status::Ok();
}

Status SchedulerImpl::Schedule(ExecutionPlan* plan) {
  if (plan == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Scheduler::Schedule: plan must not be null");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "Scheduler::Schedule: scheduler is not initialized");
  }
  *plan = ExecutionPlan{};
  plan->step_id = next_step_id_++;
  if (ActiveRequestCount() == 0) {
    return Status::Ok();
  }
  const Status status = recompute_attention_
                            ? ScheduleRecompute(plan)
                            : ScheduleWithKv(plan);
  if (!status.ok()) {
    plan->entries.clear();
    plan->total_tokens = 0;
  }
  return status;
}

Status SchedulerImpl::ScheduleRecompute(ExecutionPlan* plan) {
  I32 remaining = max_num_tokens_;
  I32 scheduled = 0;
  for (auto& entry : requests_) {
    RequestState& state = entry.second;
    if (!IsActiveStatus(state.status)) {
      continue;
    }
    if (remaining <= 0) {
      break;
    }
    const I64 history_size =
        static_cast<I64>(state.sequence.history_token_ids.size());
    if (history_size <= 0) {
      return MakeStatus(ErrorCode::kInternal,
                        "Scheduler::Schedule: empty history");
    }
    if (history_size > static_cast<I64>(max_num_tokens_)) {
      return MakeStatus(
          ErrorCode::kInternal,
          "Scheduler::Schedule: history exceeds recompute budget");
    }
    if (history_size > static_cast<I64>(remaining)) {
      continue;
    }

    PlanEntry plan_entry;
    plan_entry.request_id = state.request_id;
    plan_entry.phase = state.sequence.generated_count == 0
                           ? RequestPhase::kPrefill
                           : RequestPhase::kDecode;
    plan_entry.packed_offset = plan->total_tokens;
    plan_entry.token_count = ClampToI32(history_size);
    plan_entry.token_ids = state.sequence.history_token_ids;
    plan_entry.positions.resize(plan_entry.token_ids.size());
    for (Usize index = 0; index < plan_entry.positions.size(); ++index) {
      plan_entry.positions[index] = static_cast<I64>(index);
    }
    plan_entry.context_length = plan_entry.token_count - 1;
    plan_entry.output_slot =
        plan_entry.packed_offset + plan_entry.token_count - 1;
    plan->entries.push_back(std::move(plan_entry));
    plan->total_tokens += ClampToI32(history_size);
    remaining -= ClampToI32(history_size);
    state.computed_tokens = ClampToI32(history_size);
    state.status = RequestStatus::kRunning;
    ++scheduled;
  }
  if (scheduled == 0) {
    return MakeStatus(ErrorCode::kInternal,
                      "Scheduler::Schedule: no request fits the budget");
  }
  return Status::Ok();
}

Status SchedulerImpl::ScheduleWithKv(ExecutionPlan* plan) {
  std::vector<U64> waiting_ids;
  std::vector<U64> running_ids;
  for (const auto& entry : requests_) {
    if (entry.second.status == RequestStatus::kWaiting ||
        entry.second.status == RequestStatus::kChunkedPrefill) {
      waiting_ids.push_back(entry.first);
    } else if (entry.second.status == RequestStatus::kRunning) {
      running_ids.push_back(entry.first);
    }
  }

  I32 remaining = max_num_tokens_;
  Bool reserve_failed = false;
  Status first_reserve_error = Status::Ok();

  for (U64 request_id : waiting_ids) {
    if (remaining <= 0) {
      break;
    }
    RequestState& state = requests_.at(request_id);
    const I64 prompt_remaining =
        static_cast<I64>(state.sequence.prompt_count) -
        static_cast<I64>(state.computed_tokens);
    if (prompt_remaining <= 0) {
      state.status = RequestStatus::kRunning;
      continue;
    }
    const I64 chunk =
        std::min(prompt_remaining,
                 static_cast<I64>(std::min(max_prefill_chunk_, remaining)));
    if (chunk <= 0) {
      continue;
    }
    const Status reserve_status =
        kv_allocator_->Reserve(request_id, chunk);
    if (!reserve_status.ok()) {
      if (!reserve_failed) {
        reserve_failed = true;
        first_reserve_error = reserve_status;
      }
      continue;
    }

    PlanEntry plan_entry;
    plan_entry.request_id = request_id;
    plan_entry.phase = RequestPhase::kPrefill;
    plan_entry.packed_offset = plan->total_tokens;
    plan_entry.token_count = ClampToI32(chunk);
    const I32 first_token = state.computed_tokens;
    plan_entry.token_ids.assign(
        state.sequence.history_token_ids.begin() + first_token,
        state.sequence.history_token_ids.begin() + first_token +
            plan_entry.token_count);
    plan_entry.positions.resize(plan_entry.token_ids.size());
    for (Usize index = 0; index < plan_entry.positions.size(); ++index) {
      plan_entry.positions[index] =
          static_cast<I64>(first_token) + static_cast<I64>(index);
    }
    Status status =
        kv_allocator_->GetBlockTable(request_id, &plan_entry.block_table);
    if (!status.ok()) {
      return PrefixStatus(status, "Scheduler::Schedule");
    }
    status = kv_allocator_->GetSlots(request_id, first_token, chunk,
                                     &plan_entry.slots);
    if (!status.ok()) {
      return PrefixStatus(status, "Scheduler::Schedule");
    }
    plan_entry.context_length = first_token;
    const Bool completes_prompt =
        static_cast<I64>(first_token) + chunk ==
        static_cast<I64>(state.sequence.prompt_count);
    plan_entry.is_chunk = !completes_prompt;
    plan_entry.output_slot =
        completes_prompt
            ? plan_entry.packed_offset + plan_entry.token_count - 1
            : -1;

    const I32 entry_tokens = plan_entry.token_count;
    plan->entries.push_back(std::move(plan_entry));
    plan->total_tokens += entry_tokens;
    remaining -= entry_tokens;
    state.computed_tokens += entry_tokens;
    state.kv_reserved = true;
    state.status = completes_prompt ? RequestStatus::kRunning
                                    : RequestStatus::kChunkedPrefill;
  }

  for (U64 request_id : running_ids) {
    if (remaining <= 0) {
      break;
    }
    RequestState& state = requests_.at(request_id);
    const I64 history_size =
        static_cast<I64>(state.sequence.history_token_ids.size());
    if (static_cast<I64>(state.computed_tokens) >= history_size) {
      continue;
    }
    const Status reserve_status = kv_allocator_->Reserve(request_id, 1);
    if (!reserve_status.ok()) {
      if (!reserve_failed) {
        reserve_failed = true;
        first_reserve_error = reserve_status;
      }
      continue;
    }

    PlanEntry plan_entry;
    plan_entry.request_id = request_id;
    plan_entry.phase = RequestPhase::kDecode;
    plan_entry.packed_offset = plan->total_tokens;
    plan_entry.token_count = 1;
    plan_entry.token_ids = {
        state.sequence.history_token_ids.at(
            static_cast<Usize>(state.computed_tokens))};
    plan_entry.positions = {static_cast<I64>(state.computed_tokens)};
    Status status =
        kv_allocator_->GetBlockTable(request_id, &plan_entry.block_table);
    if (!status.ok()) {
      return PrefixStatus(status, "Scheduler::Schedule");
    }
    status = kv_allocator_->GetSlots(request_id, state.computed_tokens, 1,
                                     &plan_entry.slots);
    if (!status.ok()) {
      return PrefixStatus(status, "Scheduler::Schedule");
    }
    plan_entry.context_length = state.computed_tokens;
    plan_entry.output_slot = plan_entry.packed_offset;
    plan_entry.is_chunk = false;

    plan->entries.push_back(std::move(plan_entry));
    plan->total_tokens += 1;
    remaining -= 1;
    state.computed_tokens += 1;
    state.kv_reserved = true;
    state.status = RequestStatus::kRunning;
  }

  if (plan->entries.empty() && reserve_failed) {
    return first_reserve_error;
  }
  return Status::Ok();
}

Status SchedulerImpl::Update(U64 request_id,
                             const std::vector<Token>& tokens,
                             Response* response) {
  if (response == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Scheduler::Update: response must not be null");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "Scheduler::Update: scheduler is not initialized");
  }
  auto iterator = requests_.find(request_id);
  if (iterator == requests_.end()) {
    return MakeStatus(ErrorCode::kRequestNotFound,
                      "Scheduler::Update: request not found");
  }
  RequestState& state = iterator->second;
  if (IsTerminalStatus(state.status)) {
    FillResponse(state, state.status, response);
    return Status::Ok();
  }

  for (const Token& token : tokens) {
    if (token.token_id >= 0 && vocab_size_ > 0 &&
        token.token_id >= static_cast<I64>(vocab_size_)) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "Scheduler::Update: token id is out of vocab");
    }
  }

  Bool appended = false;
  Bool stopped = false;
  if (state.status == RequestStatus::kRunning) {
    for (const Token& token : tokens) {
      if (token.token_id < 0) {
        continue;
      }
      state.sequence.current_token = token;
      state.sequence.history_token_ids.push_back(token.token_id);
      ++state.sequence.generated_count;
      appended = true;
      if (ShouldStop(state)) {
        stopped = true;
        break;
      }
    }
  }

  if (stopped) {
    state.status = RequestStatus::kStopped;
    ReleaseIfReserved(&state);
  } else if (appended) {
    state.status = RequestStatus::kRunning;
  }
  FillResponse(state, state.status, response);
  return Status::Ok();
}

Status SchedulerImpl::Stop(U64 request_id, Response* response) {
  if (response == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Scheduler::Stop: response must not be null");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "Scheduler::Stop: scheduler is not initialized");
  }
  auto iterator = requests_.find(request_id);
  if (iterator == requests_.end()) {
    return MakeStatus(ErrorCode::kRequestNotFound,
                      "Scheduler::Stop: request not found");
  }
  RequestState& state = iterator->second;
  if (!IsTerminalStatus(state.status)) {
    state.status = RequestStatus::kStopped;
    ReleaseIfReserved(&state);
  }
  FillResponse(state, state.status, response);
  return Status::Ok();
}

Status SchedulerImpl::Fail(U64 request_id, ErrorCode code,
                           Response* response) {
  if (response == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Scheduler::Fail: response must not be null");
  }
  if (code == ErrorCode::kOk) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Scheduler::Fail: error code must not be kOk");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "Scheduler::Fail: scheduler is not initialized");
  }
  auto iterator = requests_.find(request_id);
  if (iterator == requests_.end()) {
    return MakeStatus(ErrorCode::kRequestNotFound,
                      "Scheduler::Fail: request not found");
  }
  RequestState& state = iterator->second;
  if (!IsTerminalStatus(state.status)) {
    state.status = RequestStatus::kIllegal;
    ReleaseIfReserved(&state);
  }
  FillResponse(state, state.status, response);
  return Status::Ok();
}

std::vector<RequestSnapshot> SchedulerImpl::Snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<RequestSnapshot> snapshots;
  snapshots.reserve(requests_.size());
  for (const auto& entry : requests_) {
    const RequestState& state = entry.second;
    RequestSnapshot snapshot;
    snapshot.request_id = state.request_id;
    snapshot.status = state.status;
    snapshot.prompt_tokens = state.sequence.prompt_count;
    snapshot.computed_tokens = state.computed_tokens;
    snapshot.generated_tokens = state.sequence.generated_count;
    snapshots.push_back(snapshot);
  }
  return snapshots;
}

SchedulerStats SchedulerImpl::Stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  SchedulerStats stats;
  for (const auto& entry : requests_) {
    const RequestStatus status = entry.second.status;
    if (status == RequestStatus::kWaiting ||
        status == RequestStatus::kChunkedPrefill) {
      ++stats.waiting_requests;
    } else if (status == RequestStatus::kRunning) {
      ++stats.running_requests;
    }
  }
  if (kv_allocator_ != nullptr) {
    stats.free_blocks = kv_allocator_->FreeBlocks();
    const I32 block_size = kv_allocator_->BlockSize();
    stats.total_blocks =
        block_size > 0 ? kv_allocator_->MaxTotalTokens() / block_size : 0;
  }
  return stats;
}

Bool SchedulerImpl::HasRequests() const {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& entry : requests_) {
    if (IsActiveStatus(entry.second.status)) {
      return true;
    }
  }
  return false;
}

I64 SchedulerImpl::EffectiveMaxTokens(const RequestState& state) const {
  I64 limit = state.sample_param.max_tokens > 0
                  ? static_cast<I64>(state.sample_param.max_tokens)
                  : context_length_ -
                        static_cast<I64>(state.sequence.prompt_count);
  const I64 context_room =
      context_length_ - static_cast<I64>(state.sequence.prompt_count);
  if (context_room > 0) {
    limit = std::min(limit, context_room);
  }
  if (recompute_attention_) {
    const I64 budget_room =
        static_cast<I64>(max_num_tokens_) -
        static_cast<I64>(state.sequence.prompt_count);
    if (budget_room > 0) {
      limit = std::min(limit, budget_room);
    }
  }
  return limit;
}

Bool SchedulerImpl::ShouldStop(const RequestState& state) const {
  const I64 generated =
      static_cast<I64>(state.sequence.generated_count);
  const I64 max_tokens = EffectiveMaxTokens(state);
  if (max_tokens <= 0 || generated >= max_tokens) {
    return true;
  }
  const I64 history_size =
      static_cast<I64>(state.sequence.history_token_ids.size());
  if (history_size >= context_length_) {
    return true;
  }
  if (recompute_attention_ &&
      history_size >= static_cast<I64>(max_num_tokens_)) {
    return true;
  }
  if (generated < static_cast<I64>(state.sample_param.min_tokens)) {
    return false;
  }
  const I64 token_id = state.sequence.current_token.token_id;
  if (ContainsToken(state.sample_param.stop_token_ids, token_id)) {
    return true;
  }
  if (!state.sample_param.ignore_eos &&
      ContainsToken(default_stop_ids_, token_id)) {
    return true;
  }
  return false;
}

void SchedulerImpl::ReleaseIfReserved(RequestState* state) {
  if (!state->kv_reserved || kv_allocator_ == nullptr) {
    return;
  }
  kv_allocator_->Release(state->request_id);
  state->kv_reserved = false;
}

I32 SchedulerImpl::ActiveRequestCount() const {
  I32 count = 0;
  for (const auto& entry : requests_) {
    if (IsActiveStatus(entry.second.status)) {
      ++count;
    }
  }
  return count;
}

}  // namespace

std::unique_ptr<IScheduler> CreateDefaultScheduler() {
  return std::make_unique<SchedulerImpl>();
}

}  // namespace runtime
}  // namespace mobilenet
