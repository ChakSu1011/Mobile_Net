// 模块：Runtime Engine 实现
//
// 职责：
//   实现 Engine 的生命周期状态机、模块装配、加载编排、单步推理循环、I/O
//   绑定、请求转发与资源销毁；具体调度、KV、执行与后端逻辑全部委托给统一
//   接口实现。
//
// 边界：
//   - 不实现调度/KV/执行/后端算法；
//   - 不直接调用 OpEnv/Kernel，也不直接持有 Backend；
//   - 不接触文本/Tokenizer。
//
// 依赖：
//   Runtime/engine/engine.h、engine_modules.h、各模块接口、Graph 公共接口。

#include "Runtime/engine/engine.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Graph/include/attr.h"
#include "Graph/include/graph.h"
#include "Graph/include/op.h"
#include "Graph/include/storage.h"
#include "Graph/include/tensor.h"
#include "Runtime/backend/backend.h"
#include "Runtime/common/model_config.h"
#include "Runtime/common/resource_stack.h"
#include "Runtime/common/status.h"
#include "Runtime/destroy/destroyer.h"
#include "Runtime/engine/engine_modules.h"
#include "Runtime/executor/executor.h"
#include "Runtime/kvcache/kv_cache_manager.h"
#include "Runtime/loader/loader.h"
#include "Runtime/scheduler/scheduler.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace runtime {
namespace {

// KV 预算自动推导时为运行时保留的余量（字节）。
constexpr U64 kKvBudgetReserveBytes = 64ull * 1024 * 1024;

// 返回单调时钟纳秒时间戳。
//
// 返回：
//   当前 steady_clock 时间（纳秒），用于 Step 统计。
I64 NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// 比较两个 Shape 的逻辑维度是否一致。
//
// 参数：
//   left、right：待比较的 shape。
// 返回：
//   max_dim 与各逻辑维度完全一致时返回 true。
Bool ShapeEquals(const runtime_graph::Shape& left,
                 const runtime_graph::Shape& right) {
  if (left.max_dim != right.max_dim) {
    return false;
  }
  for (I64 index = 0; index < left.max_dim; ++index) {
    if (left.dims[static_cast<Usize>(index)] !=
        right.dims[static_cast<Usize>(index)]) {
      return false;
    }
  }
  return true;
}

// 校验 Executor 传入的主机 Storage 是否与图侧张量匹配。
//
// 参数：
//   expected：图侧输入/输出张量的 storage（由 Loader 绑定）；
//   host：Executor 提供的主机 storage；
//   error：失败时写入可读原因。
// 返回：
//   true 表示 device_type/指针/精度/布局/字节数/形状均合法。
Bool ValidateHostStorage(const runtime_graph::Storage& expected,
                         const runtime_graph::Storage& host,
                         std::string* error) {
  if (host.device_type != runtime_graph::DeviceType::kHost) {
    *error = "host storage device_type must be kHost";
    return false;
  }
  if (host.data_ptr == nullptr) {
    *error = "host storage data_ptr must not be null";
    return false;
  }
  if (expected.byte_size == 0) {
    *error = "graph tensor byte_size is not set";
    return false;
  }
  if (host.byte_size == 0) {
    *error = "host storage byte_size must be positive";
    return false;
  }
  if (host.precision != expected.precision) {
    *error = "host storage precision mismatch";
    return false;
  }
  if (host.layout != expected.layout) {
    *error = "host storage layout mismatch";
    return false;
  }
  if (expected.byte_size != 0 && host.byte_size != expected.byte_size) {
    *error = "host storage byte_size mismatch";
    return false;
  }
  if (host.shape.max_dim > 0 && expected.shape.max_dim > 0 &&
      !ShapeEquals(host.shape, expected.shape)) {
    *error = "host storage shape mismatch";
    return false;
  }
  return true;
}

// 由 RuntimeConfig 与 ModelConfig 生成 Scheduler 配置。
//
// 参数：
//   runtime：运行参数；
//   model：模型配置（来自 manifest）；
//   config：输出参数。
// 返回：
//   成功返回 kOk；max_num_tokens/context_length 非法返回 kInvalidArgument。
Status BuildSchedulerConfig(const RuntimeConfig& runtime,
                            const ModelConfig& model,
                            SchedulerConfig* config) {
  if (config == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "scheduler config output must not be null");
  }
  I32 max_num_tokens = runtime.max_num_tokens > 0
                               ? runtime.max_num_tokens
                               : model.max_num_tokens;
  if (max_num_tokens <= 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "max_num_tokens must be positive");
  }
  if (model.max_num_tokens > 0 && max_num_tokens > model.max_num_tokens) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "max_num_tokens exceeds graph planning limit: " +
            std::to_string(model.max_num_tokens));
  }
  I32 max_concurrent =
      runtime.max_concurrent_requests > 0
          ? runtime.max_concurrent_requests
          : std::min(max_num_tokens, 8);
  if (max_concurrent <= 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "max_concurrent_requests must be positive");
  }
  if (model.context_length <= 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "model context_length must be positive");
  }
  config->max_num_tokens = max_num_tokens;
  config->max_concurrent_requests = max_concurrent;
  config->max_prefill_chunk = max_num_tokens;
  config->context_length = model.context_length;
  config->stop_ids = model.stop_token_ids;
  config->recompute_attention = runtime.recompute_attention;
  return Status::Ok();
}

// 由运行参数、模型配置与后端能力生成 KV 规格。
//
// 参数：
//   runtime：运行参数（可覆盖 block size）；
//   model：模型配置；
//   capability：后端能力（提供 attention_block_size）；
//   spec：输出参数。
// 返回：
//   成功返回 kOk；块大小/模型字段/精度非法返回 kInvalidArgument。
Status BuildKvCacheSpec(const RuntimeConfig& runtime,
                        const ModelConfig& model,
                        const BackendCapability& capability,
                        KvCacheSpec* spec) {
  if (spec == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "kv cache spec output must not be null");
  }
  I32 block_size = runtime.kv_block_size > 0
                           ? runtime.kv_block_size
                           : capability.attention_block_size;
  if (block_size <= 0) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "KV block size is not declared by backend or runtime config");
  }
  if (model.num_layers <= 0 || model.num_kv_heads <= 0 ||
      model.head_dim <= 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "model KV shape fields must be positive");
  }
  if (model.dtype == DataType::kUnknown) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "model dtype must not be unknown");
  }
  spec->type = KvCacheType::kFullAttention;
  spec->block_size = block_size;
  spec->num_layers = model.num_layers;
  spec->num_kv_heads = model.num_kv_heads;
  spec->head_dim = model.head_dim;
  spec->dtype = model.dtype;
  spec->sliding_window = 0;
  return Status::Ok();
}

// 计算 KV 预算。
//
// 规则：
//   - kv_cache_bytes > 0 时直接使用；
//   - 否则由 memory_budget_bytes 减去权重/arena/IO 与保留余量；
//   - 保留余量固定为 64 MiB，供运行时缓冲与碎片使用；
//   - 两者均为 0 或预算不足返回错误。
Status ComputeKvBudget(const RuntimeConfig& runtime,
                       const LoadResult& result, U64* budget) {
  if (budget == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "kv budget output must not be null");
  }
  if (runtime.kv_cache_bytes > 0) {
    *budget = runtime.kv_cache_bytes;
    return Status::Ok();
  }
  if (runtime.memory_budget_bytes == 0) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "kv_cache_bytes or memory_budget_bytes must be configured");
  }
  const U64 known_bytes =
      result.weights_bytes + result.arena_bytes + result.io_bytes;
  if (runtime.memory_budget_bytes <= known_bytes) {
    return MakeStatus(ErrorCode::kDeviceOom,
                      "memory budget does not cover weights/arena/io");
  }
  const U64 remaining = runtime.memory_budget_bytes - known_bytes;
  if (remaining <= kKvBudgetReserveBytes) {
    return MakeStatus(ErrorCode::kDeviceOom,
                      "memory budget leaves no room for KV cache");
  }
  *budget = remaining - kKvBudgetReserveBytes;
  return Status::Ok();
}

// 判断错误码是否属于设备级致命错误。
//
// 参数：
//   code：错误码。
// 返回：
//   设备/内核/缓冲/后端初始化错误返回 true；请求级错误返回 false。
Bool IsFatalError(ErrorCode code) {
  switch (code) {
    case ErrorCode::kDeviceError:
    case ErrorCode::kKernelError:
    case ErrorCode::kBufferError:
    case ErrorCode::kBackendInitError:
      return true;
    default:
      return false;
  }
}

}  // namespace

Engine::Engine() : resource_stack_(std::make_unique<ResourceStack>()) {}

Engine::~Engine() {
  if (state() != EngineState::kDestroyed) {
    (void)Destroy();
  }
}

Status Engine::SetModules(EngineModules modules) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (state_ != EngineState::kCreated) {
    return MakeStatus(ErrorCode::kStateError,
                      "Engine::SetModules requires CREATED state");
  }
  if (modules.loader) {
    loader_ = std::move(modules.loader);
  }
  if (modules.scheduler) {
    scheduler_ = std::move(modules.scheduler);
  }
  if (modules.executor) {
    executor_ = std::move(modules.executor);
  }
  if (modules.destroyer) {
    destroyer_ = std::move(modules.destroyer);
  }
  return Status::Ok();
}

Status Engine::EnsureModules() {
  if (!resource_stack_) {
    resource_stack_ = std::make_unique<ResourceStack>();
  }
  if (!destroyer_) {
    destroyer_ = CreateDefaultDestroyer();
    if (!destroyer_) {
      return MakeStatus(ErrorCode::kUnsupportedOp,
                        "default destroyer is not available");
    }
  }
  if (!loader_) {
    loader_ = CreateDefaultLoader();
    if (!loader_) {
      return MakeStatus(ErrorCode::kUnsupportedOp,
                        "default loader is not available");
    }
  }
  if (!scheduler_) {
    scheduler_ = CreateDefaultScheduler();
    if (!scheduler_) {
      return MakeStatus(ErrorCode::kUnsupportedOp,
                        "default scheduler is not available");
    }
  }
  if (!executor_) {
    executor_ = CreateDefaultExecutor();
    if (!executor_) {
      return MakeStatus(ErrorCode::kUnsupportedOp,
                        "default executor is not available");
    }
  }
  if (!kv_cache_) {
    kv_cache_ = CreateDefaultKVCache();
    if (!kv_cache_) {
      return MakeStatus(ErrorCode::kUnsupportedOp,
                        "default kv cache is not available");
    }
  }
  return Status::Ok();
}

Status Engine::Load(const EngineConfig& config) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != EngineState::kCreated) {
      return MakeStatus(ErrorCode::kStateError,
                        "Engine::Load requires CREATED state");
    }
  }
  if (config.package_dir.empty()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "package_dir must not be empty");
  }
  Status status = EnsureModules();
  if (!status.ok()) {
    return status;
  }
  if (resource_stack_->size() != 0) {
    return MakeStatus(ErrorCode::kStateError,
                      "resource stack is not empty before load");
  }

  Bool backend_initialized = false;
  Bool executor_initialized = false;
  LoadResult result;

  // 加载失败回滚：与正常 Destroy 共用 Destroyer 与 ResourceStack。
  auto rollback = [&](Backend* pending_backend, Bool backend_ready,
                      Bool executor_ready,
                      const Status& cause) -> Status {
    DestroyContext context;
    context.scheduler = scheduler_.get();
    context.kv_cache = kv_cache_.get();
    context.executor = executor_.get();
    context.graph = graph_.get();
    context.backend = executor_ready && executor_
                          ? executor_->backend()
                          : pending_backend;
    context.resource_stack = resource_stack_.get();
    context.backend_initialized = backend_ready;
    context.executor_initialized = executor_ready;
    const Status destroy_status =
        destroyer_ ? destroyer_->Destroy(context) : Status::Ok();
    graph_.reset();
    resource_stack_ = std::make_unique<ResourceStack>();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      state_ = EngineState::kCreated;
      backend_initialized_ = false;
      executor_initialized_ = false;
      input_ops_.clear();
      output_ops_.clear();
    }
    if (!destroy_status.ok()) {
      return Status(cause.code(), cause.message() +
                                      "; rollback failed: " +
                                      destroy_status.message());
    }
    return PrefixStatus(cause, "Engine::Load rollback");
  };

  LoadConfig load_config;
  load_config.package_dir = config.package_dir;
  load_config.runtime = config.runtime;
  status = loader_->Load(load_config, resource_stack_.get(), &result);
  if (!status.ok()) {
    return rollback(result.backend.get(), result.backend != nullptr, false,
                    status);
  }
  if (!result.graph || !result.backend) {
    return rollback(result.backend.get(), result.backend != nullptr, false,
                    MakeStatus(ErrorCode::kInternal,
                               "loader returned empty graph or backend"));
  }

  graph_ = std::move(result.graph);
  backend_initialized = true;

  SchedulerConfig scheduler_config;
  status = BuildSchedulerConfig(config.runtime, result.model,
                                &scheduler_config);
  if (!status.ok()) {
    return rollback(result.backend.get(), backend_initialized,
                    executor_initialized, status);
  }

  BackendCapability capability;
  status = result.backend->Capability(&capability);
  if (!status.ok()) {
    return rollback(result.backend.get(), backend_initialized,
                    executor_initialized, status);
  }

  KvCacheSpec kv_spec;
  status = BuildKvCacheSpec(config.runtime, result.model, capability, &kv_spec);
  if (!status.ok()) {
    return rollback(result.backend.get(), backend_initialized,
                    executor_initialized, status);
  }

  U64 kv_budget = 0;
  status = ComputeKvBudget(config.runtime, result, &kv_budget);
  if (!status.ok()) {
    return rollback(result.backend.get(), backend_initialized,
                    executor_initialized, status);
  }

  status = kv_cache_->Init(result.backend.get(), kv_spec, kv_budget,
                           resource_stack_.get());
  if (!status.ok()) {
    return rollback(result.backend.get(), backend_initialized,
                    executor_initialized, status);
  }

  status = scheduler_->Init(scheduler_config, kv_cache_.get(), result.model);
  if (!status.ok()) {
    return rollback(result.backend.get(), backend_initialized,
                    executor_initialized, status);
  }

  status = executor_->Init(std::move(result.backend), graph_.get(), this);
  if (!status.ok()) {
    // IExecutor::Init 失败时必须自行释放已接管的 Backend；此处不再让
    // Destroyer 触碰已经由 Executor 释放的 Backend。
    return rollback(nullptr, false, false, status);
  }
  executor_initialized = true;

  status = BindGraphIo();
  if (!status.ok()) {
    return rollback(nullptr, backend_initialized, executor_initialized,
                    status);
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    runtime_config_ = config.runtime;
    model_config_ = result.model;
    backend_initialized_ = backend_initialized;
    executor_initialized_ = executor_initialized;
    fatal_.store(false);
    fatal_code_.store(ErrorCode::kOk);
    stats_ = EngineStats{};
    state_ = EngineState::kLoaded;
  }
  return Status::Ok();
}

Status Engine::Start() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (state_ != EngineState::kLoaded) {
    return MakeStatus(ErrorCode::kStateError,
                      "Engine::Start requires LOADED state");
  }
  if (runtime_config_.threaded) {
    stop_requested_.store(false);
    try {
      inference_thread_ = std::thread(&Engine::InferenceLoop, this);
    } catch (const std::exception&) {
      return MakeStatus(ErrorCode::kInternal,
                        "failed to start inference thread");
    }
  }
  state_ = EngineState::kRunning;
  return Status::Ok();
}

Status Engine::AddRequest(const Request& request, U64* request_id) {
  if (request_id == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "request_id output must not be null");
  }
  if (request.prompt_token_ids.empty()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "prompt_token_ids must not be empty");
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != EngineState::kLoaded && state_ != EngineState::kRunning) {
      return MakeStatus(ErrorCode::kStateError,
                        "Engine::AddRequest requires LOADED/RUNNING state");
    }
    if (fatal_.load()) {
      return MakeStatus(fatal_code_.load(), "engine is in fatal state");
    }
  }
  if (!scheduler_) {
    return MakeStatus(ErrorCode::kStateError, "scheduler is not available");
  }
  Status status = scheduler_->Merge(request, request_id);
  if (status.ok()) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++stats_.total_requests;
    }
    idle_cv_.notify_all();
  }
  return status;
}

Status Engine::CancelRequest(U64 request_id) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != EngineState::kLoaded && state_ != EngineState::kRunning) {
      return MakeStatus(ErrorCode::kStateError,
                        "Engine::CancelRequest requires LOADED/RUNNING state");
    }
  }
  if (!scheduler_) {
    return MakeStatus(ErrorCode::kStateError, "scheduler is not available");
  }
  Response response;
  Status status = scheduler_->Stop(request_id, &response);
  if (status.ok() && response.request_id != 0) {
    NotifyResponse(response);
  }
  idle_cv_.notify_all();
  return status;
}

Status Engine::Update(U64 request_id,
                      const std::vector<Token>& tokens,
                      Response* response) {
  if (response == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "response must not be null");
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != EngineState::kLoaded && state_ != EngineState::kRunning) {
      return MakeStatus(ErrorCode::kStateError,
                        "Engine::Update requires LOADED/RUNNING state");
    }
  }
  if (!scheduler_) {
    return MakeStatus(ErrorCode::kStateError, "scheduler is not available");
  }
  Status update_status = scheduler_->Update(request_id, tokens, response);
  if (!update_status.ok()) {
    return update_status;
  }
  if (!tokens.empty()) {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.total_tokens += tokens.size();
  }
  NotifyResponse(*response);
  return Status::Ok();
}

Status Engine::RunOnce(StepStats* stats) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != EngineState::kLoaded && state_ != EngineState::kRunning) {
      return MakeStatus(ErrorCode::kStateError,
                        "Engine::RunOnce requires LOADED/RUNNING state");
    }
    if (runtime_config_.threaded) {
      return MakeStatus(ErrorCode::kStateError,
                        "Engine::RunOnce is disabled in threaded mode");
    }
    if (fatal_.load()) {
      return MakeStatus(fatal_code_.load(), "engine is in fatal state");
    }
  }
  return ExecuteStep(stats);
}

Status Engine::ExecuteStep(StepStats* stats) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != EngineState::kLoaded && state_ != EngineState::kRunning) {
      return MakeStatus(ErrorCode::kStateError,
                        "Engine::ExecuteStep requires LOADED/RUNNING state");
    }
    if (fatal_.load()) {
      return MakeStatus(fatal_code_.load(), "engine is in fatal state");
    }
  }
  if (!scheduler_ || !kv_cache_ || !executor_) {
    return MakeStatus(ErrorCode::kStateError,
                      "scheduler/kv_cache/executor is not available");
  }

  // 将请求标记失败并通知 Service；失败路径不阻塞其他请求。
  auto fail_request = [this](U64 request_id, ErrorCode code) {
    Response response;
    Status status = scheduler_->Fail(request_id, code, &response);
    if (status.ok() && response.request_id != 0) {
      NotifyResponse(response);
    }
  };

  const I64 schedule_start_ns = NowNs();
  ExecutionPlan plan;
  Status status = scheduler_->Schedule(&plan);
  const I64 schedule_ns = NowNs() - schedule_start_ns;
  if (!status.ok()) {
    if (IsFatalError(status.code())) {
      fatal_.store(true);
      fatal_code_.store(status.code());
    }
    return status;
  }

  StepResult step_result;
  I64 execute_ns = 0;
  if (!plan.entries.empty()) {
    KvCacheView kv_view;
    status = kv_cache_->BuildView(&kv_view);
    if (!status.ok()) {
      for (const PlanEntry& entry : plan.entries) {
        fail_request(entry.request_id, status.code());
      }
      if (IsFatalError(status.code())) {
        fatal_.store(true);
        fatal_code_.store(status.code());
      }
      return status;
    }
    const I64 execute_start_ns = NowNs();
    status = executor_->Execute(plan, kv_view, &step_result);
    execute_ns = NowNs() - execute_start_ns;
    if (!status.ok()) {
      for (const PlanEntry& entry : plan.entries) {
        fail_request(entry.request_id, status.code());
      }
      if (IsFatalError(status.code())) {
        fatal_.store(true);
        fatal_code_.store(status.code());
      }
      return status;
    }
  }

  std::unordered_map<U64, const RequestStepResult*> result_by_id;
  for (const RequestStepResult& result : step_result.requests) {
    result_by_id.emplace(result.request_id, &result);
  }

  std::unordered_set<U64> handled_requests;
  I32 generated_tokens = 0;
  for (const PlanEntry& entry : plan.entries) {
    if (!handled_requests.insert(entry.request_id).second) {
      continue;
    }
    auto result_it = result_by_id.find(entry.request_id);
    if (result_it == result_by_id.end()) {
      // 纯 Prefill chunk 不产生 token（output_slot<0），Executor 不返回结果。
      if (entry.output_slot < 0) {
        continue;
      }
      fail_request(entry.request_id, ErrorCode::kInternal);
      continue;
    }
    const RequestStepResult& result = *result_it->second;
    if (!result.ok) {
      fail_request(result.request_id, result.error);
      continue;
    }
    if (entry.output_slot < 0) {
      // 该 entry 不产出 token，忽略 Executor 可能返回的占位 token。
      continue;
    }
    Response response;
    status = scheduler_->Update(result.request_id, {result.token}, &response);
    if (!status.ok()) {
      if (status.code() == ErrorCode::kRequestNotFound) {
        continue;
      }
      return status;
    }
    ++generated_tokens;
    NotifyResponse(response);
  }

  StepStats step_stats;
  step_stats.step_id = plan.step_id;
  step_stats.total_tokens = plan.total_tokens;
  for (const PlanEntry& entry : plan.entries) {
    if (entry.phase == RequestPhase::kPrefill) {
      step_stats.prefill_tokens += entry.token_count;
    } else {
      step_stats.decode_tokens += entry.token_count;
    }
  }
  const SchedulerStats scheduler_stats = scheduler_->Stats();
  step_stats.running_requests = scheduler_stats.running_requests;
  step_stats.waiting_requests = scheduler_stats.waiting_requests;
  step_stats.preempted_requests = scheduler_stats.preempted_requests;
  const KvCacheStats kv_stats = kv_cache_->Stats();
  step_stats.kv_used_blocks = kv_stats.total_blocks - kv_stats.free_blocks;
  step_stats.kv_free_blocks = kv_stats.free_blocks;
  step_stats.schedule_ns = schedule_ns;
  step_stats.execute_ns = execute_ns;
  if (stats != nullptr) {
    *stats = step_stats;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.total_steps += 1;
    stats_.total_tokens += static_cast<U64>(generated_tokens);
    stats_.running_requests = step_stats.running_requests;
    stats_.waiting_requests = step_stats.waiting_requests;
    stats_.kv_used_blocks = step_stats.kv_used_blocks;
    stats_.kv_free_blocks = step_stats.kv_free_blocks;
    stats_.last_step_ns = schedule_ns + execute_ns;
  }
  return Status::Ok();
}

void Engine::InferenceLoop() {
  while (!stop_requested_.load()) {
    StepStats stats;
    Status status = ExecuteStep(&stats);
    if (!status.ok() &&
        (fatal_.load() || status.code() == ErrorCode::kStateError)) {
      break;
    }
    const Bool has_requests = scheduler_ && scheduler_->HasRequests();
    if (!has_requests) {
      std::unique_lock<std::mutex> lock(mutex_);
      idle_cv_.notify_all();
      idle_cv_.wait_for(lock, std::chrono::milliseconds(10), [this] {
        return stop_requested_.load() ||
               (scheduler_ && scheduler_->HasRequests());
      });
    } else if (!status.ok()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  idle_cv_.notify_all();
}

Status Engine::RunUntilIdle(U32 timeout_ms) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != EngineState::kLoaded && state_ != EngineState::kRunning) {
      return MakeStatus(
          ErrorCode::kStateError,
          "Engine::RunUntilIdle requires LOADED/RUNNING state");
    }
    if (fatal_.load()) {
      return MakeStatus(fatal_code_.load(), "engine is in fatal state");
    }
  }
  if (!scheduler_) {
    return MakeStatus(ErrorCode::kStateError, "scheduler is not available");
  }
  if (!scheduler_->HasRequests()) {
    return Status::Ok();
  }
  if (timeout_ms == 0) {
    return MakeStatus(ErrorCode::kTimeout,
                      "requests are still pending and timeout is zero");
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  if (runtime_config_.threaded) {
    std::unique_lock<std::mutex> lock(mutex_);
    while (scheduler_ && scheduler_->HasRequests()) {
      if (idle_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
        return MakeStatus(ErrorCode::kTimeout,
                          "Engine::RunUntilIdle timed out");
      }
    }
    return Status::Ok();
  }
  while (scheduler_->HasRequests()) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return MakeStatus(ErrorCode::kTimeout, "Engine::RunUntilIdle timed out");
    }
    StepStats stats;
    Status status = ExecuteStep(&stats);
    if (!status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

Status Engine::Destroy() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == EngineState::kDestroyed) {
      return Status::Ok();
    }
    if (state_ == EngineState::kDestroying) {
      return MakeStatus(ErrorCode::kStateError,
                        "Engine is already being destroyed");
    }
    if (inference_thread_.joinable() &&
        std::this_thread::get_id() == inference_thread_.get_id()) {
      return MakeStatus(ErrorCode::kStateError,
                        "Engine::Destroy must not run on the inference thread");
    }
    state_ = EngineState::kDestroying;
    stop_requested_.store(true);
  }
  idle_cv_.notify_all();
  if (inference_thread_.joinable()) {
    inference_thread_.join();
  }

  DestroyContext context;
  context.scheduler = scheduler_.get();
  context.kv_cache = kv_cache_.get();
  context.executor = executor_.get();
  context.graph = graph_.get();
  context.backend = executor_initialized_ && executor_
                        ? executor_->backend()
                        : nullptr;
  context.resource_stack = resource_stack_.get();
  context.backend_initialized = backend_initialized_;
  context.executor_initialized = executor_initialized_;
  Status status = destroyer_ ? destroyer_->Destroy(context) : Status::Ok();

  {
    std::lock_guard<std::mutex> lock(mutex_);
    graph_.reset();
    executor_.reset();
    scheduler_.reset();
    kv_cache_.reset();
    loader_.reset();
    destroyer_.reset();
    resource_stack_.reset();
    input_ops_.clear();
    output_ops_.clear();
    backend_initialized_ = false;
    executor_initialized_ = false;
    state_ = EngineState::kDestroyed;
  }
  return status;
}

void Engine::SetResponseCallback(ResponseCallback callback) {
  std::lock_guard<std::mutex> lock(mutex_);
  response_callback_ = std::move(callback);
}

void Engine::NotifyResponse(const Response& response) {
  ResponseCallback callback;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    callback = response_callback_;
  }
  if (!callback) {
    return;
  }
  try {
    callback(response);
  } catch (const std::exception&) {
    // 回调异常不得穿透 Engine；丢弃本次通知。
  }
}

Status Engine::SetInput(const std::string& name,
                        const runtime_graph::Storage& host_storage) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != EngineState::kLoaded && state_ != EngineState::kRunning) {
      return MakeStatus(ErrorCode::kStateError,
                        "Engine::SetInput requires LOADED/RUNNING state");
    }
  }
  auto it = input_ops_.find(name);
  if (it == input_ops_.end() || it->second == nullptr ||
      it->second->DstTensors().size() != 1) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "unknown input op: " + name);
  }
  const runtime_graph::Storage& expected =
      it->second->DstTensors()[0]->GetStorage();
  std::string error;
  if (!ValidateHostStorage(expected, host_storage, &error)) {
    return MakeStatus(ErrorCode::kInvalidArgument, name + ": " + error);
  }
  return it->second->SetAttr("storage", host_storage);
}

Status Engine::SetOutput(const std::string& name,
                         const runtime_graph::Storage& host_storage) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != EngineState::kLoaded && state_ != EngineState::kRunning) {
      return MakeStatus(ErrorCode::kStateError,
                        "Engine::SetOutput requires LOADED/RUNNING state");
    }
  }
  auto it = output_ops_.find(name);
  if (it == output_ops_.end() || it->second == nullptr ||
      it->second->SrcTensors().size() != 1) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "unknown output op: " + name);
  }
  const runtime_graph::Storage& expected =
      it->second->SrcTensors()[0]->GetStorage();
  std::string error;
  if (!ValidateHostStorage(expected, host_storage, &error)) {
    return MakeStatus(ErrorCode::kInvalidArgument, name + ": " + error);
  }
  return it->second->SetAttr("storage", host_storage);
}

Status Engine::BindGraphIo() {
  if (!graph_) {
    return MakeStatus(ErrorCode::kStateError, "graph is not available");
  }
  input_ops_.clear();
  output_ops_.clear();
  for (runtime_graph::Op* op : graph_->GetAllInputOp()) {
    if (op == nullptr || op->DstTensors().size() != 1) {
      return MakeStatus(ErrorCode::kInvalidModelFile,
                        "invalid input op binding");
    }
    input_ops_.emplace(op->Name(), op);
  }
  for (runtime_graph::Op* op : graph_->GetAllOutputOp()) {
    if (op == nullptr || op->SrcTensors().size() != 1) {
      return MakeStatus(ErrorCode::kInvalidModelFile,
                        "invalid output op binding");
    }
    output_ops_.emplace(op->Name(), op);
  }
  return Status::Ok();
}

EngineState Engine::state() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

EngineStats Engine::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

}  // namespace runtime
}  // namespace mobilenet
