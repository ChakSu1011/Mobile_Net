// 模块：Runtime Engine 生命周期与单步执行单元测试
//
// 职责：
//   使用 Fake Loader/Scheduler/Executor/Destroyer/KVCache/Backend 验证 Engine
//   的加载成功路径、I/O 绑定、请求入队、单步执行、回调、统计与幂等销毁；
//   同时覆盖无模块、未加载、已销毁等错误路径。
//
// 边界：
//   - 不验证真实模型包、真实调度/KV/后端算法与数值正确性；
//   - threaded 推理线程未覆盖（需要更复杂的并发 Fake）；
//   - 默认工厂桩只为本测试提供最小实现，真实模块落地后替换。
//
// 依赖：
//   Runtime/engine/engine.cc、Runtime/common/resource_stack.cc、Graph 静态库。

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Graph/include/attr.h"
#include "Graph/include/graph.h"
#include "Graph/include/op_type.h"
#include "Graph/include/storage.h"
#include "Graph/include/tensor.h"
#include "Runtime/backend/backend.h"
#include "Runtime/common/kv_view.h"
#include "Runtime/destroy/destroyer.h"
#include "Runtime/engine/engine.h"
#include "Runtime/engine/engine_modules.h"
#include "Runtime/executor/executor.h"
#include "Runtime/kvcache/kv_cache_manager.h"
#include "Runtime/loader/loader.h"
#include "Runtime/scheduler/scheduler.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace runtime {

// 测试用设备缓冲：持有主机内存，模拟设备地址。
class FakeBuffer final : public Buffer {
 public:
  // 构造指定字节数的缓冲。
  explicit FakeBuffer(U64 bytes)
      : bytes_(static_cast<Usize>(bytes)) {}

  // 返回缓冲地址；空缓冲返回 nullptr。
  void* DevicePtr() const override {
    return bytes_.empty() ? nullptr : bytes_.data();
  }

  // 返回缓冲字节数。
  U64 Bytes() const override { return bytes_.size(); }

 private:
  // Buffer::DevicePtr 为 const 方法但返回可写地址，故数据成员声明为
  // mutable，模拟设备内存的"逻辑只读、物理可写"语义。
  mutable std::vector<U8> bytes_;
};

// 测试用 Kernel：始终成功，不执行真实计算。
class FakeKernel final : public Kernel {
 public:
  // 执行一次空计算。
  Status Execute(const runtime_graph::KernelContext&) override {
    return Status::Ok();
  }
};

// 测试用 Backend：能力声明、缓冲、传输、Kernel 与同步均为最小实现。
class FakeBackend final : public Backend {
 public:
  // 初始化后端并记录状态。
  Status Init(const BackendConfig&) override {
    initialized_ = true;
    return Status::Ok();
  }

  // 反初始化后端并累计次数。
  Status Deinit() override {
    initialized_ = false;
    ++deinit_count_;
    return Status::Ok();
  }

  // 返回测试能力：FP16/INT64、H2D/D2H、16 token KV 块。
  Status Capability(BackendCapability* capability) const override {
    if (capability == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument, "capability is null");
    }
    capability->supported_dtypes = {DataType::kInt64, DataType::kFloat16};
    capability->supported_ops = {
        runtime_graph::OpType::kInput, runtime_graph::OpType::kAdd,
        runtime_graph::OpType::kSample, runtime_graph::OpType::kOutput};
    capability->h2d = true;
    capability->d2h = true;
    capability->max_buffer_bytes = 1ull << 30;
    capability->min_alignment = 256;
    capability->attention_block_size = 16;
    capability->kv_layout = KvLayout::kNhd;
    return Status::Ok();
  }

  // 创建主机模拟设备缓冲。
  Status CreateBuffer(const BufferDesc& desc,
                      std::unique_ptr<Buffer>* buffer) override {
    if (buffer == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument, "buffer is null");
    }
    *buffer = std::make_unique<FakeBuffer>(desc.bytes);
    return Status::Ok();
  }

  // 销毁缓冲（由 unique_ptr 完成）。
  Status DestroyBuffer(std::unique_ptr<Buffer> buffer) override {
    (void)buffer;
    return Status::Ok();
  }

  // 转换并创建常驻缓冲（测试不做真实转换）。
  Status ConvertParam(const ParamDesc& desc,
                      std::unique_ptr<Buffer>* buffer) override {
    if (buffer == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument, "buffer is null");
    }
    *buffer = std::make_unique<FakeBuffer>(desc.bytes);
    return Status::Ok();
  }

  // 主机到设备拷贝。
  Status Upload(void* device_ptr, const void* host_ptr, U64 bytes,
                Stream* stream) override {
    (void)stream;
    if (bytes > 0) {
      std::memcpy(device_ptr, host_ptr, static_cast<Usize>(bytes));
    }
    return Status::Ok();
  }

  // 设备到主机拷贝。
  Status Download(void* host_ptr, const void* device_ptr, U64 bytes,
                  Stream* stream) override {
    (void)stream;
    if (bytes > 0) {
      std::memcpy(host_ptr, device_ptr, static_cast<Usize>(bytes));
    }
    return Status::Ok();
  }

  // 创建测试 Kernel。
  Status CreateKernel(const KernelDesc& desc,
                      std::unique_ptr<Kernel>* kernel) override {
    (void)desc;
    if (kernel == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument, "kernel is null");
    }
    *kernel = std::make_unique<FakeKernel>();
    return Status::Ok();
  }

  // 销毁 Kernel（由 unique_ptr 完成）。
  Status DestroyKernel(std::unique_ptr<Kernel> kernel) override {
    (void)kernel;
    return Status::Ok();
  }

  // 同步执行流。
  Status Synchronize(Stream* stream) override {
    (void)stream;
    return Status::Ok();
  }

  // 返回 Deinit 调用次数。
  I32 deinit_count() const { return deinit_count_; }

 private:
  Bool initialized_ = false;
  I32 deinit_count_ = 0;
};

// 测试用 KVCache：记录规格并返回固定视图，不做真实块分配。
class FakeKVCache final : public IKVCache {
 public:
  // 保存规格。
  Status Init(Backend* backend, const KvCacheSpec& spec,
              U64 kv_budget_bytes,
              ResourceStack* resource_stack) override {
    (void)backend;
    (void)kv_budget_bytes;
    (void)resource_stack;
    spec_ = spec;
    return Status::Ok();
  }

  // 返回固定视图。
  Status BuildView(KvCacheView* view) const override {
    if (view == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument, "view is null");
    }
    view->layers.assign(1, KvLayerView{});
    view->block_size = BlockSize();
    view->max_total_tokens = MaxTotalTokens();
    return Status::Ok();
  }

  // 记录追加分配。
  Status Reserve(U64 request_id, I64 new_tokens) override {
    (void)request_id;
    if (new_tokens <= 0) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "new_tokens must be positive");
    }
    return Status::Ok();
  }

  // 释放请求（测试无状态）。
  void Release(U64 request_id) override { (void)request_id; }

  // 返回固定空闲块数。
  I64 FreeBlocks() const override { return 16; }

  // 返回块大小。
  I32 BlockSize() const override {
    return spec_.block_size > 0 ? spec_.block_size : 16;
  }

  // 返回固定最大 token 数。
  I64 MaxTotalTokens() const override { return 64; }

  // 返回固定块表。
  Status GetBlockTable(U64 request_id,
                       std::vector<I32>* table) const override {
    (void)request_id;
    if (table == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument, "table is null");
    }
    table->assign(1, 0);
    return Status::Ok();
  }

  // 返回固定 slot 列表。
  Status GetSlots(U64 request_id, I64 first_token, I64 count,
                  std::vector<I64>* slots) const override {
    (void)request_id;
    (void)first_token;
    if (slots == nullptr || count < 0) {
      return MakeStatus(ErrorCode::kInvalidArgument, "invalid slots query");
    }
    slots->assign(static_cast<Usize>(count), 0);
    return Status::Ok();
  }

  // 返回固定 KV 统计。
  KvCacheStats Stats() const override {
    KvCacheStats stats;
    stats.block_size = BlockSize();
    stats.total_blocks = 16;
    stats.free_blocks = 16;
    stats.max_total_tokens = MaxTotalTokens();
    return stats;
  }

 private:
  KvCacheSpec spec_;
};

// 测试用 Scheduler：单请求单步、Update 后立即结束。
class FakeScheduler final : public IScheduler {
 public:
  // 保存配置与 KV 接口。
  Status Init(const SchedulerConfig& config, KvAllocator* kv_allocator,
              const ModelConfig& model) override {
    config_ = config;
    kv_allocator_ = kv_allocator;
    model_ = model;
    return Status::Ok();
  }

  // 接收请求并分配 ID。
  Status Merge(const Request& request, U64* request_id) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (request_id == nullptr || request.prompt_token_ids.empty()) {
      return MakeStatus(ErrorCode::kInvalidArgument, "invalid request");
    }
    RequestData data;
    data.request_id = next_request_id_++;
    data.state = RequestStatus::kWaiting;
    data.prompt_tokens =
        static_cast<I32>(request.prompt_token_ids.size());
    data.computed_tokens = 0;
    data.generated_tokens = 0;
    data.sample_param = request.sample_param;
    data.sequence.sequence_id = data.request_id;
    data.sequence.prompt_count = data.prompt_tokens;
    data.sequence.history_token_ids = request.prompt_token_ids;
    data.sequence.vocab_size = 100;
    requests_.emplace(data.request_id, data);
    *request_id = data.request_id;
    return Status::Ok();
  }

  // 生成单请求单 token 的计划。
  Status Schedule(ExecutionPlan* plan) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (plan == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument, "plan is null");
    }
    plan->entries.clear();
    plan->total_tokens = 0;
    plan->step_id = ++step_id_;
    for (auto& entry : requests_) {
      RequestData& data = entry.second;
      if (data.state != RequestStatus::kWaiting &&
          data.state != RequestStatus::kRunning) {
        continue;
      }
      PlanEntry plan_entry;
      plan_entry.request_id = data.request_id;
      plan_entry.phase = RequestPhase::kDecode;
      plan_entry.packed_offset = 0;
      plan_entry.token_count = 1;
      plan_entry.token_ids = {1};
      plan_entry.positions = {data.computed_tokens};
      plan_entry.slots = {0};
      plan_entry.block_table = {0};
      plan_entry.context_length = data.computed_tokens;
      plan_entry.output_slot = 0;
      plan_entry.is_chunk = false;
      plan->entries.push_back(plan_entry);
      plan->total_tokens += 1;
      data.state = RequestStatus::kRunning;
      data.computed_tokens += 1;
      break;
    }
    return Status::Ok();
  }

  // 回写 token、更新 Sequence 并结束请求。
  Status Update(U64 request_id, const std::vector<Token>& tokens,
                Response* response) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = requests_.find(request_id);
    if (it == requests_.end()) {
      return MakeStatus(ErrorCode::kRequestNotFound, "request not found");
    }
    if (response == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument, "response is null");
    }
    RequestData& data = it->second;
    for (const Token& token : tokens) {
      data.sequence.history_token_ids.push_back(token.token_id);
      data.sequence.current_token = token;
      ++data.sequence.generated_count;
      ++data.generated_tokens;
    }
    data.state = RequestStatus::kStopped;
    FillResponse(data, response);
    return Status::Ok();
  }

  // 取消请求并生成终态 Response。
  Status Stop(U64 request_id, Response* response) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = requests_.find(request_id);
    if (it == requests_.end()) {
      return MakeStatus(ErrorCode::kRequestNotFound, "request not found");
    }
    if (response == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument, "response is null");
    }
    it->second.state = RequestStatus::kStopped;
    FillResponse(it->second, response);
    return Status::Ok();
  }

  // 标记请求失败并生成终态 Response。
  Status Fail(U64 request_id, ErrorCode code,
              Response* response) override {
    (void)code;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = requests_.find(request_id);
    if (it == requests_.end()) {
      return MakeStatus(ErrorCode::kRequestNotFound, "request not found");
    }
    if (response == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument, "response is null");
    }
    it->second.state = RequestStatus::kIllegal;
    FillResponse(it->second, response);
    return Status::Ok();
  }

  // 返回请求快照。
  std::vector<RequestSnapshot> Snapshot() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<RequestSnapshot> snapshots;
    for (const auto& entry : requests_) {
      RequestSnapshot snapshot;
      snapshot.request_id = entry.second.request_id;
      snapshot.status = entry.second.state;
      snapshot.prompt_tokens = entry.second.prompt_tokens;
      snapshot.computed_tokens = entry.second.computed_tokens;
      snapshot.generated_tokens = entry.second.generated_tokens;
      snapshots.push_back(snapshot);
    }
    return snapshots;
  }

  // 返回调度统计。
  SchedulerStats Stats() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    SchedulerStats stats;
    for (const auto& entry : requests_) {
      if (entry.second.state == RequestStatus::kWaiting) {
        ++stats.waiting_requests;
      } else if (entry.second.state == RequestStatus::kRunning) {
        ++stats.running_requests;
      }
    }
    stats.total_blocks = 16;
    stats.free_blocks = 16;
    return stats;
  }

  // 是否存在未结束请求。
  Bool HasRequests() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& entry : requests_) {
      if (entry.second.state == RequestStatus::kWaiting ||
          entry.second.state == RequestStatus::kRunning) {
        return true;
      }
    }
    return false;
  }

 private:
  // 请求内部状态。
  struct RequestData {
    U64 request_id = 0;
    RequestStatus state = RequestStatus::kWaiting;
    I32 prompt_tokens = 0;
    I32 computed_tokens = 0;
    I32 generated_tokens = 0;
    Sequence sequence;
    SampleParams sample_param;
  };

  // 用内部请求状态填充 Response。
  //
  // 参数：
  //   data：内部请求状态；
  //   response：输出参数，不允许为空。
  void FillResponse(const RequestData& data, Response* response) const {
    response->request_id = data.request_id;
    response->status = data.state;
    response->sequences.assign(1, data.sequence);
    response->sample_param = data.sample_param;
  }

  mutable std::mutex mutex_;
  std::unordered_map<U64, RequestData> requests_;
  SchedulerConfig config_;
  ModelConfig model_;
  KvAllocator* kv_allocator_ = nullptr;
  U64 next_request_id_ = 1;
  U64 step_id_ = 0;
};

// 测试用 Executor：为每个计划项返回 token 42。
class FakeExecutor final : public IExecutor {
 public:
  // 接管 Backend 并保存引用。
  Status Init(std::unique_ptr<Backend> backend, runtime_graph::Graph* graph,
              IoBinder* io_binder) override {
    backend_ = std::move(backend);
    graph_ = graph;
    io_binder_ = io_binder;
    return Status::Ok();
  }

  // 返回拥有的 Backend 指针。
  Backend* backend() override { return backend_.get(); }

  // 为每个计划项返回固定 token。
  Status Execute(const ExecutionPlan& plan, const KvCacheView& kv_view,
                 StepResult* result) override {
    (void)kv_view;
    if (result == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument, "result is null");
    }
    result->requests.clear();
    for (const PlanEntry& entry : plan.entries) {
      RequestStepResult step_result;
      step_result.request_id = entry.request_id;
      step_result.token = Token{42, 0.0f, 0.0f, true};
      step_result.ok = true;
      result->requests.push_back(step_result);
    }
    return Status::Ok();
  }

  // 释放 Backend。
  Status Deinit() override {
    Status status = backend_ ? backend_->Deinit() : Status::Ok();
    backend_.reset();
    return status;
  }

 private:
  std::unique_ptr<Backend> backend_;
  runtime_graph::Graph* graph_ = nullptr;
  IoBinder* io_binder_ = nullptr;
};

// 测试用 Destroyer：清空资源栈并反初始化 Executor/Backend。
class FakeDestroyer final : public IDestroyer {
 public:
  // 按与默认 Destroyer 相同的顺序释放测试资源。
  Status Destroy(const DestroyContext& context) override {
    if (context.resource_stack != nullptr) {
      Status status = context.resource_stack->Clear();
      if (!status.ok()) {
        return status;
      }
    }
    if (context.executor != nullptr && context.executor_initialized) {
      return context.executor->Deinit();
    }
    if (context.backend != nullptr && context.backend_initialized) {
      return context.backend->Deinit();
    }
    return Status::Ok();
  }
};

// 设置张量的测试 storage（int64、row_major、[1,4]、32 字节）。
//
// 参数：
//   tensor：待设置的图张量；
//   bytes：字节数。
// 返回：
//   成功返回 kOk。
Status SetTestTensorStorage(runtime_graph::Tensor* tensor, U64 bytes) {
  if (tensor == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument, "tensor is null");
  }
  runtime_graph::Storage& storage = tensor->GetStorage();
  storage.precision = runtime_graph::Precision::kInt64;
  storage.layout = runtime_graph::Layout::kRowMajor;
  storage.device_type = runtime_graph::DeviceType::kDevice;
  storage.byte_size = static_cast<Usize>(bytes);
  storage.reusable = false;
  storage.shape.max_dim = 2;
  storage.shape.dims[0] = 1;
  storage.shape.dims[1] = 4;
  storage.shape.pitch[0] = 4;
  storage.shape.pitch[1] = 1;
  return Status::Ok();
}

// 构造测试模型配置。
ModelConfig MakeTestModelConfig() {
  ModelConfig model;
  model.num_layers = 2;
  model.num_heads = 4;
  model.num_kv_heads = 2;
  model.head_dim = 8;
  model.hidden_size = 32;
  model.intermediate_size = 64;
  model.vocab_size = 100;
  model.context_length = 64;
  model.rms_norm_eps = 1e-6f;
  model.rope_theta = 10000.0;
  model.dtype = DataType::kFloat16;
  model.max_num_tokens = 4;
  model.max_batch_size = 1;
  model.pad_token_id = 0;
  model.bos_token_id = 0;
  model.stop_token_ids = {2, 3};
  return model;
}

// 测试用 Loader：构造最小图与 FakeBackend。
class FakeLoader final : public ILoader {
 public:
  // 构造 input → add → sample → output 的最小图并返回。
  Status Load(const LoadConfig& config, ResourceStack* resource_stack,
              LoadResult* result) override {
    (void)config;
    (void)resource_stack;
    if (result == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument, "result is null");
    }
    auto graph = std::make_unique<runtime_graph::Graph>("engine_test");
    runtime_graph::Tensor* input = nullptr;
    Status status = graph->CreateInput(
        "input.input_ids", runtime_graph::Precision::kInt64,
        runtime_graph::Layout::kRowMajor, &input);
    if (!status.ok()) {
      return status;
    }
    status = SetTestTensorStorage(input, 32);
    if (!status.ok()) {
      return status;
    }
    std::vector<runtime_graph::Tensor*> add_outputs;
    status = graph->CreateOp("add", {input}, 1, runtime_graph::OpType::kAdd,
                             runtime_graph::AttrMap{}, &add_outputs);
    if (!status.ok() || add_outputs.size() != 1) {
      return status.ok() ? MakeStatus(ErrorCode::kInternal, "bad op output")
                         : status;
    }
    runtime_graph::Tensor* add_out = add_outputs[0];
    status = SetTestTensorStorage(add_out, 32);
    if (!status.ok()) {
      return status;
    }
    runtime_graph::Tensor* sample_out = nullptr;
    status = graph->CreateSample("sample", add_out, &sample_out);
    if (!status.ok()) {
      return status;
    }
    status = SetTestTensorStorage(sample_out, 32);
    if (!status.ok()) {
      return status;
    }
    status = graph->CreateOutput("output.tokens", sample_out);
    if (!status.ok()) {
      return status;
    }
    status = graph->SortOp();
    if (!status.ok()) {
      return status;
    }
    result->graph = std::move(graph);
    result->backend = std::make_unique<FakeBackend>();
    result->model = MakeTestModelConfig();
    result->weights_bytes = 1024;
    result->arena_bytes = 2048;
    result->io_bytes = 256;
    return Status::Ok();
  }
};

// 以下默认工厂桩只为本测试提供最小可用实现；真实模块落地后删除。
std::unique_ptr<ILoader> CreateDefaultLoader() { return nullptr; }
std::unique_ptr<IScheduler> CreateDefaultScheduler() { return nullptr; }
std::unique_ptr<IExecutor> CreateDefaultExecutor() { return nullptr; }
std::unique_ptr<IKVCache> CreateDefaultKVCache() {
  return std::make_unique<FakeKVCache>();
}
std::unique_ptr<IDestroyer> CreateDefaultDestroyer() { return nullptr; }

}  // namespace runtime
}  // namespace mobilenet

namespace {

// 断言计数。
I32 g_failures = 0;

// 检查布尔条件。
void Check(Bool condition, const C8* message) {
  if (!condition) {
    std::printf("FAIL: %s\n", message);
    ++g_failures;
  }
}

// 检查 Status 错误码。
void CheckStatus(const mobilenet::Status& status,
                 mobilenet::ErrorCode expected, const C8* message) {
  Check(status.code() == expected, message);
}

}  // namespace

// Engine 测试入口。
I32 main() {
  using mobilenet::ErrorCode;
  using mobilenet::runtime::Engine;
  using mobilenet::runtime::EngineConfig;
  using mobilenet::runtime::EngineModules;
  using mobilenet::runtime::EngineState;
  using mobilenet::runtime::FakeDestroyer;
  using mobilenet::runtime::FakeExecutor;
  using mobilenet::runtime::FakeLoader;
  using mobilenet::runtime::FakeScheduler;
  using mobilenet::runtime::Request;
  using mobilenet::runtime::RequestStatus;
  using mobilenet::runtime::Response;
  using mobilenet::runtime::Token;
  using mobilenet::runtime_graph::DeviceType;
  using mobilenet::runtime_graph::Layout;
  using mobilenet::runtime_graph::Precision;
  using mobilenet::runtime_graph::Storage;

  // 生命周期与错误路径。
  {
    Engine engine;
    Check(engine.state() == EngineState::kCreated, "initial state is CREATED");
    CheckStatus(engine.SetModules(EngineModules{}), ErrorCode::kOk,
                "SetModules accepts empty modules");
    EngineConfig empty_config;
    CheckStatus(engine.Load(empty_config), ErrorCode::kInvalidArgument,
                "Load rejects empty package_dir");
    CheckStatus(engine.RunOnce(nullptr), ErrorCode::kStateError,
                "RunOnce rejects CREATED state");
    Request request;
    request.prompt_token_ids = {1, 2, 3};
    U64 request_id = 0;
    CheckStatus(engine.AddRequest(request, &request_id),
                ErrorCode::kStateError,
                "AddRequest rejects CREATED state");
    CheckStatus(engine.CancelRequest(1), ErrorCode::kStateError,
                "CancelRequest rejects CREATED state");
    Response response;
    CheckStatus(engine.Update(1, {Token{1, 0.0f, 0.0f, true}}, &response),
                ErrorCode::kStateError, "Update rejects CREATED state");
    CheckStatus(engine.RunUntilIdle(1), ErrorCode::kStateError,
                "RunUntilIdle rejects CREATED state");
    Storage storage;
    CheckStatus(engine.SetInput("input.input_ids", storage),
                ErrorCode::kStateError, "SetInput rejects CREATED state");
    CheckStatus(engine.SetOutput("output.tokens", storage),
                ErrorCode::kStateError, "SetOutput rejects CREATED state");
    CheckStatus(engine.Destroy(), ErrorCode::kOk,
                "Destroy succeeds from CREATED");
    Check(engine.state() == EngineState::kDestroyed,
          "state becomes DESTROYED");
    CheckStatus(engine.Destroy(), ErrorCode::kOk, "Destroy is idempotent");
  }

  // 默认模块缺失路径。
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

  // 加载成功、I/O 绑定、单步执行与销毁。
  {
    Engine engine;
    EngineModules modules;
    modules.loader = std::make_unique<FakeLoader>();
    modules.scheduler = std::make_unique<FakeScheduler>();
    modules.executor = std::make_unique<FakeExecutor>();
    modules.destroyer = std::make_unique<FakeDestroyer>();
    CheckStatus(engine.SetModules(std::move(modules)), ErrorCode::kOk,
                "SetModules accepts fake modules");

    EngineConfig config;
    config.package_dir = "fake-package";
    config.runtime.kv_cache_bytes = 1ull << 20;
    config.runtime.max_num_tokens = 4;
    config.runtime.max_concurrent_requests = 2;
    config.runtime.kv_block_size = 0;
    config.runtime.threaded = false;
    CheckStatus(engine.Load(config), ErrorCode::kOk,
                "Load succeeds with fake modules");
    Check(engine.state() == EngineState::kLoaded, "state becomes LOADED");

    std::vector<I64> input_data(4, 1);
    Storage input_storage;
    input_storage.precision = Precision::kInt64;
    input_storage.layout = Layout::kRowMajor;
    input_storage.device_type = DeviceType::kHost;
    input_storage.byte_size = input_data.size() * sizeof(I64);
    input_storage.data_ptr = input_data.data();
    input_storage.shape.max_dim = 2;
    input_storage.shape.dims[0] = 1;
    input_storage.shape.dims[1] = 4;
    CheckStatus(engine.SetInput("input.input_ids", input_storage),
                ErrorCode::kOk, "SetInput succeeds");

    std::vector<I64> output_data(4, 0);
    Storage output_storage;
    output_storage.precision = Precision::kInt64;
    output_storage.layout = Layout::kRowMajor;
    output_storage.device_type = DeviceType::kHost;
    output_storage.byte_size = output_data.size() * sizeof(I64);
    output_storage.data_ptr = output_data.data();
    output_storage.shape.max_dim = 2;
    output_storage.shape.dims[0] = 1;
    output_storage.shape.dims[1] = 4;
    CheckStatus(engine.SetOutput("output.tokens", output_storage),
                ErrorCode::kOk, "SetOutput succeeds");

    I64 callback_token = -1;
    Bool callback_finished = false;
    I32 callback_count = 0;
    engine.SetResponseCallback([&](const Response& response) {
      if (!response.sequences.empty()) {
        callback_token = response.sequences[0].current_token.token_id;
      }
      callback_finished = response.status == RequestStatus::kStopped ||
                           response.status == RequestStatus::kIllegal;
      ++callback_count;
    });

    Request request;
    request.prompt_token_ids = {1, 2, 3};
    U64 request_id = 0;
    CheckStatus(engine.AddRequest(request, &request_id), ErrorCode::kOk,
                "AddRequest succeeds");
    Check(request_id != 0, "request id is assigned");

    mobilenet::runtime::StepStats step_stats;
    CheckStatus(engine.RunOnce(&step_stats), ErrorCode::kOk,
                "RunOnce succeeds");
    Check(callback_count == 1 && callback_token == 42 &&
              callback_finished,
          "token callback receives finished token");
    Check(step_stats.total_tokens == 1, "step token count is 1");
    const mobilenet::runtime::EngineStats stats = engine.stats();
    Check(stats.total_steps == 1 && stats.total_tokens == 1,
          "engine stats updated");
    CheckStatus(engine.RunUntilIdle(100), ErrorCode::kOk,
                "RunUntilIdle returns when idle");
    CheckStatus(engine.Destroy(), ErrorCode::kOk,
                "Destroy succeeds after execution");
    Check(engine.state() == EngineState::kDestroyed,
          "state becomes DESTROYED after execution");
  }

  // threaded 模式：启动推理线程、等待回调、RunUntilIdle 与销毁。
  {
    Engine engine;
    EngineModules modules;
    modules.loader = std::make_unique<FakeLoader>();
    modules.scheduler = std::make_unique<FakeScheduler>();
    modules.executor = std::make_unique<FakeExecutor>();
    modules.destroyer = std::make_unique<FakeDestroyer>();
    CheckStatus(engine.SetModules(std::move(modules)), ErrorCode::kOk,
                "threaded SetModules succeeds");

    EngineConfig config;
    config.package_dir = "fake-package";
    config.runtime.kv_cache_bytes = 1ull << 20;
    config.runtime.max_num_tokens = 4;
    config.runtime.max_concurrent_requests = 2;
    config.runtime.kv_block_size = 0;
    config.runtime.threaded = true;
    CheckStatus(engine.Load(config), ErrorCode::kOk,
                "threaded Load succeeds");
    CheckStatus(engine.Start(), ErrorCode::kOk, "threaded Start succeeds");
    CheckStatus(engine.RunOnce(nullptr), ErrorCode::kStateError,
                "RunOnce is disabled in threaded mode");

    std::atomic<I64> callback_token{-1};
    std::atomic<Bool> callback_finished{false};
    std::atomic<I32> callback_count{0};
    engine.SetResponseCallback([&](const Response& response) {
      if (!response.sequences.empty()) {
        callback_token.store(response.sequences[0].current_token.token_id);
      }
      callback_finished.store(response.status == RequestStatus::kStopped ||
                              response.status == RequestStatus::kIllegal);
      callback_count.fetch_add(1);
    });

    Request request;
    request.prompt_token_ids = {1, 2, 3};
    U64 request_id = 0;
    CheckStatus(engine.AddRequest(request, &request_id), ErrorCode::kOk,
                "threaded AddRequest succeeds");
    for (I32 i = 0; i < 2000 && !callback_finished.load(); ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    Check(callback_count.load() == 1 && callback_token.load() == 42 &&
              callback_finished.load(),
          "threaded callback receives finished token");
    CheckStatus(engine.RunUntilIdle(1000), ErrorCode::kOk,
                "threaded RunUntilIdle succeeds");
    Check(engine.stats().total_steps >= 1,
          "threaded engine executed at least one step");
    CheckStatus(engine.Destroy(), ErrorCode::kOk,
                "threaded Destroy succeeds");
    Check(engine.state() == EngineState::kDestroyed,
          "threaded state becomes DESTROYED");
  }

  if (g_failures != 0) {
    std::printf("%d engine checks failed\n", g_failures);
    return 1;
  }
  std::printf("all engine checks passed\n");
  return 0;
}
