// 模块：Runtime Destroy 单元测试
//
// 职责：
//   用真实 Graph + Fake Backend/Executor 验证默认 Destroyer 的释放顺序、
//   OpEnv/缓冲/GraphEnv 清理、Executor 优先于 Backend 的反初始化、错误收集
//   与重复调用幂等。
//
// 边界：
//   - Fake Backend/Executor 只记录调用，不做真实设备操作；
//   - 不覆盖推理线程停止（由 Engine 负责）与 KVCache 内部块释放；
//   - 不验证设备资源真实回收。
//
// 依赖：
//   Runtime/destroy/destroyer.cc、Runtime/common/{status,resource_stack}.cc、
//   Graph 静态库。

#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "Graph/include/attr.h"
#include "Graph/include/graph.h"
#include "Graph/include/graph_env.h"
#include "Graph/include/op.h"
#include "Graph/include/op_env.h"
#include "Graph/include/op_type.h"
#include "Graph/include/tensor.h"
#include "Runtime/backend/backend.h"
#include "Runtime/common/io_binder.h"
#include "Runtime/common/resource_stack.h"
#include "Runtime/common/status.h"
#include "Runtime/destroy/destroyer.h"
#include "Runtime/executor/executor.h"
#include "Utils/cpp/types.h"

namespace {

using mobilenet::Backend;
using mobilenet::BackendCapability;
using mobilenet::BackendConfig;
using mobilenet::Buffer;
using mobilenet::BufferDesc;
using mobilenet::DataType;
using mobilenet::ErrorCode;
using mobilenet::ErrorCodeName;
using mobilenet::KernelDesc;
using mobilenet::MakeStatus;
using mobilenet::ParamDesc;
using mobilenet::Status;
using mobilenet::Stream;
using mobilenet::runtime::CreateDefaultDestroyer;
using mobilenet::runtime::DestroyContext;
using mobilenet::runtime::IDestroyer;
using mobilenet::runtime::IExecutor;
using mobilenet::runtime::IoBinder;
using mobilenet::runtime::ResourceStack;
using mobilenet::runtime_graph::AttrMap;
using mobilenet::runtime_graph::Graph;
using mobilenet::runtime_graph::GraphEnv;
using mobilenet::runtime_graph::Op;
using mobilenet::runtime_graph::OpEnv;
using mobilenet::runtime_graph::OpType;
using mobilenet::runtime_graph::Tensor;

// 全局断言计数器。
I32 g_checks = 0;
I32 g_failures = 0;

// 记录一条断言结果。
//
// 参数：
//   condition：断言条件；
//   message：断言描述。
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

// 返回字符串在日志中首次出现的下标。
//
// 参数：
//   log：日志；
//   item：待查找项。
// 返回：
//   下标；不存在返回 -1。
I32 IndexOf(const std::vector<std::string>& log, const std::string& item) {
  for (Usize index = 0; index < log.size(); ++index) {
    if (log[index] == item) {
      return static_cast<I32>(index);
    }
  }
  return -1;
}

// 返回字符串在日志中最后一次出现的下标。
//
// 参数：
//   log：日志；
//   item：待查找项。
// 返回：
//   下标；不存在返回 -1。
I32 LastIndexOf(const std::vector<std::string>& log,
                const std::string& item) {
  for (Usize index = log.size(); index > 0; --index) {
    if (log[index - 1] == item) {
      return static_cast<I32>(index - 1);
    }
  }
  return -1;
}

// 计数型 OpEnv。
//
// 功能：
//   析构时累加计数并写日志，用于验证 Destroy 是否调用 ClearOpEnv。
class CountingOpEnv final : public OpEnv {
 public:
  // 构造。
  //
  // 参数：
  //   op：所属 op；
  //   destroy_count：析构计数输出；
  //   log：事件日志输出。
  CountingOpEnv(Op* op, I32* destroy_count,
                std::vector<std::string>* log)
      : OpEnv(op), destroy_count_(destroy_count), log_(log) {}

  // 析构并记录事件。
  ~CountingOpEnv() override {
    ++*destroy_count_;
    log_->push_back("openv");
  }

  // 空执行。
  Status Forward() override { return Status::Ok(); }

  // 不申请临时 workspace。
  Bool NeedTempData() const override { return false; }

 private:
  // 析构计数。
  I32* destroy_count_ = nullptr;

  // 事件日志。
  std::vector<std::string>* log_ = nullptr;
};

// 简单设备缓冲占位。
class FakeBuffer final : public Buffer {
 public:
  // 返回空设备地址；测试不访问内容。
  void* DevicePtr() const override { return nullptr; }

  // 返回零字节。
  U64 Bytes() const override { return 0; }
};

// Fake 后端。
//
// 功能：
//   记录 Synchronize/Deinit 调用顺序与次数，可注入失败状态。
class FakeBackend final : public Backend {
 public:
  // 初始化。
  Status Init(const BackendConfig& config) override {
    (void)config;
    return Status::Ok();
  }

  // 反初始化并记录。
  Status Deinit() override {
    ++deinit_calls;
    if (log != nullptr) {
      log->push_back("backend_deinit");
    }
    return deinit_status;
  }

  // 返回空能力。
  Status Capability(BackendCapability* capability) const override {
    if (capability == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "FakeBackend: capability must not be null");
    }
    *capability = BackendCapability{};
    return Status::Ok();
  }

  // 创建缓冲；测试不通过 Destroyer 直接分配。
  Status CreateBuffer(const BufferDesc& desc,
                      std::unique_ptr<Buffer>* buffer) override {
    (void)desc;
    if (buffer == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "FakeBackend: buffer output must not be null");
    }
    *buffer = std::make_unique<FakeBuffer>();
    return Status::Ok();
  }

  // 销毁缓冲并记录。
  Status DestroyBuffer(std::unique_ptr<Buffer> buffer) override {
    (void)buffer;
    if (log != nullptr) {
      log->push_back("backend_buffer");
    }
    return Status::Ok();
  }

  // 权重转换；测试不使用。
  Status ConvertParam(const ParamDesc& desc,
                      std::unique_ptr<Buffer>* buffer) override {
    (void)desc;
    (void)buffer;
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "FakeBackend: ConvertParam is not supported");
  }

  // 主机到设备拷贝；测试不使用。
  Status Upload(void* device_ptr, const void* host_ptr, U64 bytes,
                Stream* stream) override {
    (void)device_ptr;
    (void)host_ptr;
    (void)bytes;
    (void)stream;
    return Status::Ok();
  }

  // 设备到主机拷贝；测试不使用。
  Status Download(void* host_ptr, const void* device_ptr, U64 bytes,
                  Stream* stream) override {
    (void)host_ptr;
    (void)device_ptr;
    (void)bytes;
    (void)stream;
    return Status::Ok();
  }

  // 创建 Kernel；测试不使用。
  Status CreateKernel(const KernelDesc& desc,
                      std::unique_ptr<mobilenet::Kernel>* kernel) override {
    (void)desc;
    (void)kernel;
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "FakeBackend: CreateKernel is not supported");
  }

  // 销毁 Kernel；测试不使用。
  Status DestroyKernel(std::unique_ptr<mobilenet::Kernel> kernel) override {
    (void)kernel;
    return Status::Ok();
  }

  // 同步并记录。
  Status Synchronize(Stream* stream) override {
    (void)stream;
    ++sync_calls;
    if (log != nullptr) {
      log->push_back("sync");
    }
    return sync_status;
  }

  // 事件日志。
  std::vector<std::string>* log = nullptr;

  // Synchronize 次数。
  I32 sync_calls = 0;

  // Deinit 次数。
  I32 deinit_calls = 0;

  // Synchronize 返回状态。
  Status sync_status;

  // Deinit 返回状态。
  Status deinit_status;
};

// Fake 执行器。
//
// 功能：
//   记录 Deinit 次数与事件；backend() 返回借用后端。
class FakeExecutor final : public IExecutor {
 public:
  // 构造。
  //
  // 参数：
  //   backend：借用后端指针；
  //   log：事件日志。
  FakeExecutor(Backend* backend, std::vector<std::string>* log)
      : backend_(backend), log_(log) {}

  // 测试不走 Init。
  Status Init(std::unique_ptr<Backend> backend, Graph* graph,
              IoBinder* io_binder) override {
    (void)backend;
    (void)graph;
    (void)io_binder;
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "FakeExecutor: Init is not supported");
  }

  // 返回借用后端。
  Backend* backend() override { return backend_; }

  // 测试不走 Execute。
  Status Execute(const mobilenet::runtime::ExecutionPlan& plan,
                 const mobilenet::runtime::KvCacheView& kv_view,
                 mobilenet::runtime::StepResult* result) override {
    (void)plan;
    (void)kv_view;
    (void)result;
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "FakeExecutor: Execute is not supported");
  }

  // 反初始化并记录。
  Status Deinit() override {
    ++deinit_calls;
    if (log_ != nullptr) {
      log_->push_back("executor_deinit");
    }
    return deinit_status;
  }

  // Deinit 次数。
  I32 deinit_calls = 0;

  // Deinit 返回状态。
  Status deinit_status;

 private:
  // 借用后端。
  Backend* backend_ = nullptr;

  // 事件日志。
  std::vector<std::string>* log_ = nullptr;
};

// 构建含输入/Add/输出的测试图。
//
// 参数：
//   graph：待构建图，非空；
//   op_env_destroyed：OpEnv 析构计数；
//   log：事件日志。
// 返回：
//   成功返回 true，并为全部 op 绑定 CountingOpEnv。
Bool BuildGraph(Graph* graph, I32* op_env_destroyed,
                std::vector<std::string>* log) {
  if (graph == nullptr) {
    return false;
  }
  Tensor* input = nullptr;
  if (!graph->CreateInput("x", &input).ok() || input == nullptr) {
    return false;
  }
  std::vector<Tensor*> sources = {input, input};
  std::vector<Tensor*> outputs;
  if (!graph->CreateOp("add", sources, 1, OpType::kAdd, AttrMap{},
                       &outputs)
           .ok() ||
      outputs.size() != 1 || outputs[0] == nullptr) {
    return false;
  }
  Tensor* add_output = outputs[0];
  if (!graph->CreateOutput("out", add_output).ok()) {
    return false;
  }
  if (!graph->SortOp().ok()) {
    return false;
  }
  std::unordered_set<Op*> attached;
  auto attach = [&attached, op_env_destroyed,
                 log](const std::vector<Op*>& ops) -> Bool {
    for (Op* op : ops) {
      if (op == nullptr) {
        return false;
      }
      if (!attached.insert(op).second) {
        continue;
      }
      if (!op->SetOpEnv(
                   std::make_unique<CountingOpEnv>(op, op_env_destroyed, log))
               .ok()) {
        return false;
      }
    }
    return true;
  };
  return attach(graph->GetOrderlyOPList()) &&
         attach(graph->GetAllInputOp()) &&
         attach(graph->GetAllOutputOp());
}

// 测试正常销毁顺序与幂等。
void TestDestroyOrderAndIdempotency() {
  std::cout << "[CASE] destroy order and idempotency\n";
  Graph graph("destroy_test");
  I32 op_env_destroyed = 0;
  std::vector<std::string> log;
  Check(BuildGraph(&graph, &op_env_destroyed, &log),
        "BuildGraph creates input/add/output with OpEnv");
  Check(graph.SetGraphEnv(std::make_unique<GraphEnv>()).ok() &&
            graph.GetGraphEnv() != nullptr,
        "SetGraphEnv installs a GraphEnv");

  FakeBackend backend;
  backend.log = &log;
  FakeExecutor executor(&backend, &log);
  ResourceStack stack;
  I32 buffer_released = 0;
  stack.Push("buffer", [&buffer_released, &log]() -> Status {
    ++buffer_released;
    log.push_back("buffer");
    return Status::Ok();
  });

  DestroyContext context;
  context.executor = &executor;
  context.graph = &graph;
  context.backend = &backend;
  context.resource_stack = &stack;
  context.backend_initialized = true;
  context.executor_initialized = true;

  std::unique_ptr<IDestroyer> destroyer = CreateDefaultDestroyer();
  Check(destroyer != nullptr, "CreateDefaultDestroyer returns non-null");
  CheckStatus(destroyer->Destroy(context), ErrorCode::kOk,
              "first Destroy succeeds");
  Check(backend.sync_calls == 1, "Synchronize is called once");
  Check(executor.deinit_calls == 1, "Executor::Deinit is called once");
  Check(backend.deinit_calls == 0,
        "Backend::Deinit is not called when Executor owns it");
  Check(op_env_destroyed == 3, "all three OpEnv objects are cleared");
  Check(buffer_released == 1 && stack.size() == 0,
        "ResourceStack releases its buffer entry");
  Check(graph.GetGraphEnv() == nullptr, "GraphEnv is cleared");
  Check(IndexOf(log, "sync") >= 0 &&
            IndexOf(log, "sync") < IndexOf(log, "openv"),
        "Synchronize happens before ClearOpEnv");
  Check(LastIndexOf(log, "openv") < IndexOf(log, "buffer"),
        "OpEnv is cleared before buffered resources");
  Check(IndexOf(log, "buffer") < IndexOf(log, "executor_deinit"),
        "buffers are released before Executor::Deinit");
  Check(IndexOf(log, "backend_deinit") == -1,
        "backend Deinit is not invoked directly");

  const std::vector<std::string> saved_log = log;
  const I32 saved_sync = backend.sync_calls;
  const I32 saved_executor = executor.deinit_calls;
  const I32 saved_openv = op_env_destroyed;
  CheckStatus(destroyer->Destroy(context), ErrorCode::kOk,
              "second Destroy on the same context succeeds");
  Check(backend.sync_calls == saved_sync &&
            executor.deinit_calls == saved_executor &&
            op_env_destroyed == saved_openv && log == saved_log,
        "second Destroy is a no-op for the same context");
  context.scheduler = nullptr;
  CheckStatus(destroyer->Destroy(context), ErrorCode::kOk,
              "Destroy with explicit null scheduler succeeds");
}

// 测试未初始化 Executor 时直接反初始化 Backend。
void TestBackendPath() {
  std::cout << "[CASE] backend-only destroy path\n";
  std::vector<std::string> log;
  FakeBackend backend;
  backend.log = &log;
  ResourceStack stack;
  DestroyContext context;
  context.backend = &backend;
  context.resource_stack = &stack;
  context.backend_initialized = true;
  context.executor_initialized = false;

  std::unique_ptr<IDestroyer> destroyer = CreateDefaultDestroyer();
  CheckStatus(destroyer->Destroy(context), ErrorCode::kOk,
              "backend-only Destroy succeeds");
  Check(backend.sync_calls == 1 && backend.deinit_calls == 1,
        "Synchronize and Backend::Deinit are both called");
}

// 测试首个错误收集与后续阶段继续执行。
void TestErrorCollection() {
  std::cout << "[CASE] error collection\n";
  std::vector<std::string> log;
  FakeBackend backend;
  backend.log = &log;
  backend.sync_status =
      MakeStatus(ErrorCode::kDeviceError, "forced sync failure");
  FakeExecutor executor(&backend, &log);
  ResourceStack stack;
  I32 buffer_released = 0;
  stack.Push("buffer", [&buffer_released]() -> Status {
    ++buffer_released;
    return MakeStatus(ErrorCode::kBufferError, "forced buffer failure");
  });
  DestroyContext context;
  context.executor = &executor;
  context.backend = &backend;
  context.resource_stack = &stack;
  context.backend_initialized = true;
  context.executor_initialized = true;

  std::unique_ptr<IDestroyer> destroyer = CreateDefaultDestroyer();
  const Status status = destroyer->Destroy(context);
  CheckStatus(status, ErrorCode::kDeviceError,
              "Destroy returns the first failure (Synchronize)");
  Check(executor.deinit_calls == 1 && buffer_released == 1,
        "later release phases still run after an earlier failure");
  Check(stack.size() == 0, "ResourceStack is cleared despite failure");

  std::vector<std::string> second_log;
  FakeBackend second_backend;
  second_backend.log = &second_log;
  FakeExecutor second_executor(&second_backend, &second_log);
  second_executor.deinit_status =
      MakeStatus(ErrorCode::kBufferError, "forced executor failure");
  ResourceStack second_stack;
  DestroyContext second_context;
  second_context.executor = &second_executor;
  second_context.backend = &second_backend;
  second_context.resource_stack = &second_stack;
  second_context.backend_initialized = true;
  second_context.executor_initialized = true;
  std::unique_ptr<IDestroyer> second_destroyer = CreateDefaultDestroyer();
  CheckStatus(second_destroyer->Destroy(second_context),
              ErrorCode::kBufferError,
              "Executor::Deinit failure is propagated");
}

// 测试全空上下文与标志为真但指针为空的安全性。
void TestNullContext() {
  std::cout << "[CASE] null context\n";
  std::unique_ptr<IDestroyer> destroyer = CreateDefaultDestroyer();
  DestroyContext empty;
  CheckStatus(destroyer->Destroy(empty), ErrorCode::kOk,
              "empty DestroyContext succeeds");
  CheckStatus(destroyer->Destroy(empty), ErrorCode::kOk,
              "empty DestroyContext is idempotent");

  DestroyContext inconsistent;
  inconsistent.backend_initialized = true;
  inconsistent.executor_initialized = true;
  std::unique_ptr<IDestroyer> second_destroyer = CreateDefaultDestroyer();
  CheckStatus(second_destroyer->Destroy(inconsistent), ErrorCode::kOk,
              "initialized flags with null pointers are skipped safely");
}

}  // namespace

// 测试入口。
//
// 返回：
//   全部用例通过返回 0；存在失败断言返回 1。
int main() {
  std::cout << "Runtime Destroy unit test\n";
  TestDestroyOrderAndIdempotency();
  TestBackendPath();
  TestErrorCollection();
  TestNullContext();
  std::cout << "checks=" << g_checks << " failures=" << g_failures << "\n";
  return g_failures == 0 ? 0 : 1;
}
