// 模块：Runtime Scheduler 单元测试
//
// 职责：
//   用 Fake KvAllocator 验证默认 Scheduler 的完整历史重算、chunked prefill、
//   decode、停止判定、Stop/Fail 幂等、预算约束与错误码。
//
// 边界：
//   - 不依赖设备、Backend 或 Graph；
//   - 断言均为值级检查（计划字段、Sequence、Response、调用次数）；
//   - 不覆盖抢占与阶段三投机解码（当前实现未提供）。
//
// 依赖：
//   Runtime/scheduler/scheduler.cc、Runtime/common/{status,plan,sequence}.h。

#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "Runtime/common/kv_allocator.h"
#include "Runtime/common/model_config.h"
#include "Runtime/common/plan.h"
#include "Runtime/common/sequence.h"
#include "Runtime/common/status.h"
#include "Runtime/scheduler/scheduler.h"
#include "Utils/cpp/types.h"

namespace {

using mobilenet::ErrorCode;
using mobilenet::ErrorCodeName;
using mobilenet::MakeStatus;
using mobilenet::Status;
using mobilenet::runtime::CreateDefaultScheduler;
using mobilenet::runtime::ExecutionPlan;
using mobilenet::runtime::IScheduler;
using mobilenet::runtime::KvAllocator;
using mobilenet::runtime::ModelConfig;
using mobilenet::runtime::PlanEntry;
using mobilenet::runtime::Request;
using mobilenet::runtime::RequestPhase;
using mobilenet::runtime::RequestSnapshot;
using mobilenet::runtime::RequestStatus;
using mobilenet::runtime::Response;
using mobilenet::runtime::SchedulerConfig;
using mobilenet::runtime::SchedulerStats;
using mobilenet::runtime::Token;

// 全局断言计数器。
I32 g_checks = 0;
I32 g_failures = 0;

// 记录一条断言结果。
//
// 参数：
//   condition：断言条件；
//   message：断言描述。
// 返回：
//   无；失败时累加 g_failures 并打印。
void Check(Bool condition, const std::string& message) {
  ++g_checks;
  if (condition) {
    std::cout << "[PASS] " << message << "\n";
    return;
  }
  ++g_failures;
  std::cout << "[FAIL] " << message << "\n";
}

// 记录状态码断言。
//
// 参数：
//   status：实际状态；
//   expected：期望错误码；
//   message：断言描述。
void CheckStatus(const Status& status, ErrorCode expected,
                 const std::string& message) {
  Check(status.code() == expected,
        message + " (got " + ErrorCodeName(status.code()) + ")");
}

// Fake KV 分配器。
//
// 功能：
//   用主机侧计数器模拟块容量，记录 Reserve/Release 次数，供 Scheduler
//   断言调用路径；块表与 slot 采用简单递增编号。
//
// 边界：
//   - 不分配真实内存；
//   - fail_reserve 为 true 时 Reserve 直接返回 kDeviceOom。
class FakeKvAllocator final : public KvAllocator {
 public:
  // 构造。
  //
  // 参数：
  //   block_size：每块 token 数；
  //   total_blocks：总块数。
  FakeKvAllocator(I32 block_size, I64 total_blocks)
      : block_size_(block_size), total_blocks_(total_blocks) {}

  // 追加预留容量。
  Status Reserve(U64 request_id, I64 new_tokens) override {
    ++reserve_calls;
    if (new_tokens <= 0) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "FakeKvAllocator: new_tokens must be positive");
    }
    if (fail_reserve) {
      return MakeStatus(ErrorCode::kDeviceOom,
                        "FakeKvAllocator: forced OOM");
    }
    const auto iterator = capacities_.find(request_id);
    const I64 capacity =
        iterator == capacities_.end() ? 0 : iterator->second;
    if (capacity > MaxTotalTokens() ||
        new_tokens > MaxTotalTokens() - capacity) {
      return MakeStatus(ErrorCode::kDeviceOom,
                        "FakeKvAllocator: capacity exceeded");
    }
    capacities_[request_id] = capacity + new_tokens;
    return Status::Ok();
  }

  // 释放请求容量。
  void Release(U64 request_id) override {
    ++release_calls;
    capacities_.erase(request_id);
  }

  // 返回空闲块数。
  I64 FreeBlocks() const override {
    I64 used = 0;
    for (const auto& entry : capacities_) {
      used += BlocksFor(entry.second);
    }
    return total_blocks_ - used;
  }

  // 返回块大小。
  I32 BlockSize() const override { return block_size_; }

  // 返回最大 token 容量。
  I64 MaxTotalTokens() const override {
    return total_blocks_ * static_cast<I64>(block_size_);
  }

  // 返回块表快照。
  Status GetBlockTable(U64 request_id,
                       std::vector<I32>* table) const override {
    if (table == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "FakeKvAllocator: table must not be null");
    }
    const auto iterator = capacities_.find(request_id);
    if (iterator == capacities_.end()) {
      return MakeStatus(ErrorCode::kRequestNotFound,
                        "FakeKvAllocator: request not found");
    }
    const I64 blocks = BlocksFor(iterator->second);
    table->clear();
    for (I64 index = 0; index < blocks; ++index) {
      table->push_back(static_cast<I32>(index));
    }
    return Status::Ok();
  }

  // 计算 slot；Fake 直接使用 token 下标。
  Status GetSlots(U64 request_id, I64 first_token, I64 count,
                  std::vector<I64>* slots) const override {
    if (slots == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "FakeKvAllocator: slots must not be null");
    }
    const auto iterator = capacities_.find(request_id);
    if (iterator == capacities_.end()) {
      return MakeStatus(ErrorCode::kRequestNotFound,
                        "FakeKvAllocator: request not found");
    }
    if (first_token < 0 || count < 0 ||
        first_token > iterator->second ||
        count > iterator->second - first_token) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "FakeKvAllocator: range out of capacity");
    }
    slots->clear();
    for (I64 index = 0; index < count; ++index) {
      slots->push_back(first_token + index);
    }
    return Status::Ok();
  }

  // Reserve 调用次数。
  I32 reserve_calls = 0;

  // Release 调用次数。
  I32 release_calls = 0;

  // 是否强制 Reserve 失败。
  Bool fail_reserve = false;

 private:
  // 计算 token 容量对应的块数。
  //
  // 参数：
  //   tokens：token 容量。
  // 返回：
  //   向上取整的块数。
  I64 BlocksFor(I64 tokens) const {
    return tokens / static_cast<I64>(block_size_) +
           (tokens % static_cast<I64>(block_size_) != 0 ? 1 : 0);
  }

  // 每块 token 数。
  I32 block_size_ = 0;

  // 总块数。
  I64 total_blocks_ = 0;

  // 请求 → 已预留 token 容量。
  std::map<U64, I64> capacities_;
};

// 构造基础模型配置。
//
// 返回：
//   vocab_size=100、context_length=16、停止 token 99 的 ModelConfig。
ModelConfig MakeModel() {
  ModelConfig model;
  model.vocab_size = 100;
  model.context_length = 16;
  model.stop_token_ids = {99};
  return model;
}

// 构造基础调度配置。
//
// 参数：
//   recompute：是否启用完整历史重算；
//   max_num_tokens：单步 token 预算。
// 返回：
//   SchedulerConfig。
SchedulerConfig MakeConfig(Bool recompute, I32 max_num_tokens) {
  SchedulerConfig config;
  config.max_num_tokens = max_num_tokens;
  config.max_concurrent_requests = 8;
  config.max_prefill_chunk = max_num_tokens;
  config.context_length = 16;
  config.recompute_attention = recompute;
  return config;
}

// 构造输出 token。
//
// 参数：
//   token_id：token id。
// 返回：
//   select=true 的 Token。
Token MakeToken(I64 token_id) {
  Token token;
  token.token_id = token_id;
  token.select = true;
  return token;
}

// 测试完整历史重算的单请求流程。
//
// 覆盖：
//   Merge → Schedule（Prefill）→ Update → Schedule（Decode）→ 停止 →
//   幂等 Update/Stop；并断言重算模式不调用 KvAllocator。
void TestRecomputeSingleRequest() {
  FakeKvAllocator kv(16, 64);
  std::unique_ptr<IScheduler> scheduler = CreateDefaultScheduler();
  Check(scheduler != nullptr, "CreateDefaultScheduler returns non-null");
  Check(scheduler->Init(MakeConfig(true, 64), &kv, MakeModel()).ok(),
        "recompute Init succeeds");

  Request request;
  request.prompt_token_ids = {1, 2, 3};
  U64 request_id = 0;
  Check(scheduler->Merge(request, &request_id).ok() && request_id == 1,
        "recompute Merge assigns request_id=1");
  Check(scheduler->HasRequests(), "recompute HasRequests after Merge");

  std::vector<RequestSnapshot> snapshots = scheduler->Snapshot();
  Check(snapshots.size() == 1 &&
            snapshots[0].status == RequestStatus::kWaiting &&
            snapshots[0].prompt_tokens == 3 &&
            snapshots[0].computed_tokens == 0 &&
            snapshots[0].generated_tokens == 0,
        "recompute snapshot is Waiting with prompt=3");

  ExecutionPlan plan;
  Check(scheduler->Schedule(&plan).ok(), "recompute first Schedule succeeds");
  Check(plan.step_id == 1 && plan.total_tokens == 3 &&
            plan.entries.size() == 1,
        "recompute first plan has 1 entry and 3 tokens");
  const PlanEntry& prefill = plan.entries[0];
  Check(prefill.request_id == request_id &&
            prefill.phase == RequestPhase::kPrefill &&
            prefill.token_count == 3 && prefill.packed_offset == 0 &&
            prefill.context_length == 2 && prefill.output_slot == 2 &&
            prefill.is_chunk == false,
        "recompute Prefill entry carries full prompt metadata");
  Check(prefill.token_ids == std::vector<I64>({1, 2, 3}) &&
            prefill.positions == std::vector<I64>({0, 1, 2}),
        "recompute Prefill token ids/positions cover the full prompt");
  Check(prefill.slots.empty() && prefill.block_table.empty(),
        "recompute Prefill has empty slots/block_table");
  Check(kv.reserve_calls == 0,
        "recompute Schedule never calls KvAllocator::Reserve");

  Response response;
  Check(scheduler->Update(request_id, {MakeToken(10)}, &response).ok(),
        "recompute Update succeeds");
  Check(response.request_id == request_id &&
            response.status == RequestStatus::kRunning &&
            response.sequences.size() == 1 &&
            response.sequences[0].history_token_ids ==
                std::vector<I64>({1, 2, 3, 10}) &&
            response.sequences[0].generated_count == 1 &&
            response.sequences[0].current_token.token_id == 10,
        "recompute Update appends token and returns Running");

  ExecutionPlan decode_plan;
  Check(scheduler->Schedule(&decode_plan).ok(),
        "recompute second Schedule succeeds");
  Check(decode_plan.total_tokens == 4 &&
            decode_plan.entries.size() == 1 &&
            decode_plan.entries[0].phase == RequestPhase::kDecode &&
            decode_plan.entries[0].token_ids ==
                std::vector<I64>({1, 2, 3, 10}) &&
            decode_plan.entries[0].positions ==
                std::vector<I64>({0, 1, 2, 3}) &&
            decode_plan.entries[0].output_slot == 3 &&
            decode_plan.entries[0].context_length == 3,
        "recompute Decode replans the whole history");

  Check(scheduler->Update(request_id, {MakeToken(99)}, &response).ok() &&
            response.status == RequestStatus::kStopped &&
            response.sequences[0].generated_count == 2,
        "recompute Update stops on default stop token");
  Check(kv.release_calls == 0,
        "recompute terminal does not release KV (never reserved)");
  Check(!scheduler->HasRequests(), "recompute HasRequests false after stop");

  Check(scheduler->Update(request_id, {MakeToken(55)}, &response).ok() &&
            response.status == RequestStatus::kStopped &&
            response.sequences[0].history_token_ids.size() == 5,
        "recompute Update after terminal is idempotent");
  Check(scheduler->Stop(request_id, &response).ok() &&
            response.status == RequestStatus::kStopped,
        "recompute Stop after terminal is idempotent");
}

// 测试 KV 模式 chunked prefill 与 decode。
//
// 覆盖：
//   chunk 切分、is_chunk/output_slot/context_length、块表/slots、
//   partial chunk 的无效输出、decode、终态 Release。
void TestKvChunkedPrefill() {
  FakeKvAllocator kv(4, 16);
  std::unique_ptr<IScheduler> scheduler = CreateDefaultScheduler();
  SchedulerConfig config = MakeConfig(false, 8);
  config.max_prefill_chunk = 4;
  Check(scheduler->Init(config, &kv, MakeModel()).ok(),
        "KV Init succeeds");

  Request request;
  request.prompt_token_ids = {1, 2, 3, 4, 5, 6};
  U64 request_id = 0;
  Check(scheduler->Merge(request, &request_id).ok(), "KV Merge succeeds");

  ExecutionPlan first_plan;
  Check(scheduler->Schedule(&first_plan).ok(),
        "KV first chunk Schedule succeeds");
  Check(first_plan.total_tokens == 4 && first_plan.entries.size() == 1 &&
            first_plan.entries[0].token_count == 4 &&
            first_plan.entries[0].is_chunk &&
            first_plan.entries[0].output_slot == -1 &&
            first_plan.entries[0].context_length == 0 &&
            first_plan.entries[0].positions ==
                std::vector<I64>({0, 1, 2, 3}) &&
            first_plan.entries[0].slots.size() == 4 &&
            !first_plan.entries[0].block_table.empty(),
        "KV first chunk is 4 tokens, non-final, with slots/block_table");
  Check(kv.reserve_calls == 1, "KV first chunk reserves KV once");
  const SchedulerStats first_stats = scheduler->Stats();
  Check(first_stats.waiting_requests == 1 &&
            first_stats.running_requests == 0 &&
            first_stats.free_blocks == 15 &&
            first_stats.total_blocks == 16,
        "KV Stats reports waiting request and block usage");

  std::vector<RequestSnapshot> snapshots = scheduler->Snapshot();
  Check(snapshots.size() == 1 &&
            snapshots[0].status == RequestStatus::kChunkedPrefill &&
            snapshots[0].computed_tokens == 4,
        "KV partial prefill snapshot is ChunkedPrefill/computed=4");

  Response response;
  Check(scheduler->Update(request_id, {MakeToken(-1)}, &response).ok() &&
            response.status == RequestStatus::kChunkedPrefill &&
            response.sequences[0].history_token_ids.size() == 6 &&
            response.sequences[0].generated_count == 0,
        "KV partial chunk ignores the placeholder output token");

  ExecutionPlan second_plan;
  Check(scheduler->Schedule(&second_plan).ok(),
        "KV second chunk Schedule succeeds");
  Check(second_plan.total_tokens == 2 && second_plan.entries.size() == 1 &&
            second_plan.entries[0].token_count == 2 &&
            !second_plan.entries[0].is_chunk &&
            second_plan.entries[0].output_slot == 1 &&
            second_plan.entries[0].context_length == 4 &&
            second_plan.entries[0].positions ==
                std::vector<I64>({4, 5}),
        "KV final chunk is 2 tokens and produces output slot");
  Check(kv.reserve_calls == 2, "KV final chunk reserves KV again");

  snapshots = scheduler->Snapshot();
  Check(snapshots.size() == 1 &&
            snapshots[0].status == RequestStatus::kRunning &&
            snapshots[0].computed_tokens == 6,
        "KV completed prefill snapshot is Running/computed=6");

  Check(scheduler->Update(request_id, {MakeToken(20)}, &response).ok() &&
            response.status == RequestStatus::kRunning &&
            response.sequences[0].generated_count == 1,
        "KV prefill completion Update returns Running");

  ExecutionPlan decode_plan;
  Check(scheduler->Schedule(&decode_plan).ok(),
        "KV decode Schedule succeeds");
  Check(decode_plan.total_tokens == 1 && decode_plan.entries.size() == 1 &&
            decode_plan.entries[0].phase == RequestPhase::kDecode &&
            decode_plan.entries[0].token_ids ==
                std::vector<I64>({20}) &&
            decode_plan.entries[0].positions == std::vector<I64>({6}) &&
            decode_plan.entries[0].context_length == 6 &&
            decode_plan.entries[0].output_slot == 0 &&
            decode_plan.entries[0].slots.size() == 1,
        "KV decode entry feeds the last generated token");
  Check(kv.reserve_calls == 3, "KV decode reserves one token");

  Check(scheduler->Update(request_id, {MakeToken(99)}, &response).ok() &&
            response.status == RequestStatus::kStopped,
        "KV Update stops on stop token");
  Check(kv.release_calls == 1, "KV terminal releases allocated blocks");
  Check(!scheduler->HasRequests(), "KV HasRequests false after stop");

  ExecutionPlan empty_plan;
  Check(scheduler->Schedule(&empty_plan).ok() &&
            empty_plan.entries.empty() && empty_plan.total_tokens == 0,
        "KV Schedule with no active request returns empty plan");
}

// 测试 Stop/Fail 与参数错误码。
void TestStopFailAndInvalidArguments() {
  FakeKvAllocator kv(4, 16);
  std::unique_ptr<IScheduler> scheduler = CreateDefaultScheduler();
  Check(scheduler->Init(MakeConfig(false, 8), &kv, MakeModel()).ok(),
        "Stop/Fail Init succeeds");

  Request request;
  request.prompt_token_ids = {1, 2};
  U64 first_id = 0;
  Check(scheduler->Merge(request, &first_id).ok(), "Stop/Fail first Merge");

  Response response;
  Check(scheduler->Stop(first_id, &response).ok() &&
            response.status == RequestStatus::kStopped,
        "Stop returns Stopped response");
  Check(kv.release_calls == 0,
        "Stop before Reserve does not call Release");
  Check(scheduler->Stop(first_id, &response).ok() &&
            response.status == RequestStatus::kStopped,
        "Stop after terminal is idempotent");
  Check(scheduler->Fail(first_id, ErrorCode::kInternal, &response).ok() &&
            response.status == RequestStatus::kStopped,
        "Fail after Stop preserves the terminal state");

  Request second;
  second.prompt_token_ids = {3};
  second.request_id = 42;
  U64 second_id = 0;
  Check(scheduler->Merge(second, &second_id).ok() && second_id == 42,
        "Merge honors caller-provided request_id");
  Check(scheduler->Fail(second_id, ErrorCode::kInternal, &response).ok() &&
            response.status == RequestStatus::kIllegal,
        "Fail returns Illegal response");
  Check(scheduler->Fail(second_id, ErrorCode::kInternal, &response).ok() &&
            response.status == RequestStatus::kIllegal,
        "Fail after terminal is idempotent");
  Check(scheduler->Stop(second_id, &response).ok() &&
            response.status == RequestStatus::kIllegal,
        "Stop after Fail preserves the terminal state");

  CheckStatus(scheduler->Update(999, {MakeToken(1)}, &response),
              ErrorCode::kRequestNotFound, "Update unknown request");
  CheckStatus(scheduler->Stop(999, &response), ErrorCode::kRequestNotFound,
              "Stop unknown request");
  CheckStatus(scheduler->Fail(999, ErrorCode::kInternal, &response),
              ErrorCode::kRequestNotFound, "Fail unknown request");
  CheckStatus(scheduler->Update(first_id, {MakeToken(1)}, nullptr),
              ErrorCode::kInvalidArgument, "Update null response");
  CheckStatus(scheduler->Stop(first_id, nullptr),
              ErrorCode::kInvalidArgument, "Stop null response");
  CheckStatus(scheduler->Fail(first_id, ErrorCode::kInternal, nullptr),
              ErrorCode::kInvalidArgument, "Fail null response");
  CheckStatus(scheduler->Fail(first_id, ErrorCode::kOk, &response),
              ErrorCode::kInvalidArgument, "Fail rejects kOk code");
  CheckStatus(scheduler->Schedule(nullptr), ErrorCode::kInvalidArgument,
              "Schedule null plan");

  Request empty;
  U64 ignored_id = 0;
  CheckStatus(scheduler->Merge(empty, &ignored_id),
              ErrorCode::kInvalidArgument, "Merge rejects empty prompt");
  Request duplicate;
  duplicate.prompt_token_ids = {1};
  duplicate.request_id = first_id;
  CheckStatus(scheduler->Merge(duplicate, &ignored_id),
              ErrorCode::kInvalidArgument, "Merge rejects duplicate request_id");
  CheckStatus(scheduler->Merge(duplicate, nullptr),
              ErrorCode::kInvalidArgument, "Merge null request_id output");

  SchedulerConfig limited = MakeConfig(false, 8);
  limited.max_concurrent_requests = 1;
  std::unique_ptr<IScheduler> limited_scheduler = CreateDefaultScheduler();
  Check(limited_scheduler->Init(limited, &kv, MakeModel()).ok(),
        "limited concurrency Init succeeds");
  U64 limited_id = 0;
  Check(limited_scheduler->Merge(request, &limited_id).ok(),
        "limited concurrency first Merge succeeds");
  CheckStatus(limited_scheduler->Merge(request, &ignored_id),
              ErrorCode::kDeviceOom,
              "limited concurrency rejects second active request");
  Response limited_response;
  Check(limited_scheduler->Stop(limited_id, &limited_response).ok(),
        "limited concurrency Stop succeeds");
  Check(limited_scheduler->Merge(request, &ignored_id).ok(),
        "limited concurrency admits request after Stop");
}

// 测试重算模式的总 token 预算。
void TestRecomputeBudget() {
  FakeKvAllocator kv(4, 32);
  std::unique_ptr<IScheduler> scheduler = CreateDefaultScheduler();
  SchedulerConfig config = MakeConfig(true, 5);
  Check(scheduler->Init(config, &kv, MakeModel()).ok(),
        "budget Init succeeds");

  Request first;
  first.prompt_token_ids = {1, 2, 3};
  Request second;
  second.prompt_token_ids = {4, 5, 6};
  U64 first_id = 0;
  U64 second_id = 0;
  Check(scheduler->Merge(first, &first_id).ok() &&
            scheduler->Merge(second, &second_id).ok(),
        "budget Merge accepts two short prompts");

  ExecutionPlan plan;
  Check(scheduler->Schedule(&plan).ok(),
        "budget first Schedule succeeds");
  Check(plan.entries.size() == 1 && plan.total_tokens == 3 &&
            plan.total_tokens <= config.max_num_tokens,
        "budget first schedule stops at max_num_tokens");

  Response response;
  Check(scheduler->Update(first_id, {MakeToken(10)}, &response).ok(),
        "budget Update first request");
  Check(scheduler->Schedule(&plan).ok(),
        "budget second Schedule succeeds");
  Check(plan.entries.size() == 1 && plan.total_tokens == 4 &&
            plan.total_tokens <= config.max_num_tokens,
        "budget second schedule still respects token budget");

  Check(scheduler->Stop(first_id, &response).ok() &&
            scheduler->Stop(second_id, &response).ok(),
        "budget Stop both requests");
}

// 测试 min_tokens 与 ignore_eos。
void TestMinTokensAndIgnoreEos() {
  FakeKvAllocator kv(4, 32);
  std::unique_ptr<IScheduler> scheduler = CreateDefaultScheduler();
  Check(scheduler->Init(MakeConfig(true, 32), &kv, MakeModel()).ok(),
        "min_tokens Init succeeds");

  Request request;
  request.prompt_token_ids = {1, 2};
  request.sample_param.is_valid = true;
  request.sample_param.min_tokens = 2;
  request.sample_param.max_tokens = 4;
  request.sample_param.stop_token_ids = {7};
  U64 request_id = 0;
  Check(scheduler->Merge(request, &request_id).ok(),
        "min_tokens Merge succeeds");
  ExecutionPlan plan;
  Check(scheduler->Schedule(&plan).ok(), "min_tokens Schedule succeeds");
  Response response;
  Check(scheduler->Update(request_id, {MakeToken(7)}, &response).ok() &&
            response.status == RequestStatus::kRunning,
        "stop token is ignored before min_tokens");
  Check(scheduler->Update(request_id, {MakeToken(99)}, &response).ok() &&
            response.status == RequestStatus::kStopped,
        "default stop token stops after min_tokens");

  Request ignore_eos;
  ignore_eos.prompt_token_ids = {3};
  ignore_eos.sample_param.is_valid = true;
  ignore_eos.sample_param.max_tokens = 2;
  ignore_eos.sample_param.ignore_eos = true;
  U64 ignore_id = 0;
  Check(scheduler->Merge(ignore_eos, &ignore_id).ok(),
        "ignore_eos Merge succeeds");
  Check(scheduler->Schedule(&plan).ok(), "ignore_eos Schedule succeeds");
  Check(scheduler->Update(ignore_id, {MakeToken(99)}, &response).ok() &&
            response.status == RequestStatus::kRunning,
        "ignore_eos suppresses default stop token");
  Check(scheduler->Update(ignore_id, {MakeToken(42)}, &response).ok() &&
            response.status == RequestStatus::kStopped,
        "max_tokens stops ignore_eos request");
}

}  // namespace

int main() {
  std::cout << "Runtime Scheduler unit test\n";
  TestRecomputeSingleRequest();
  TestKvChunkedPrefill();
  TestStopFailAndInvalidArguments();
  TestRecomputeBudget();
  TestMinTokensAndIgnoreEos();
  std::cout << "checks=" << g_checks << " failures=" << g_failures << "\n";
  return g_failures == 0 ? 0 : 1;
}
