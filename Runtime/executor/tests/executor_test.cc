// 模块：Executor 单元/集成测试
//
// 职责：
//   用 MockBackend 与 C++ Graph API 构造小图
//   （Input → Gather → Linear → Sample → Output），先经默认 OpEnvFactory 挂载
//   OpEnv，再验证 IExecutor 的 Init/Execute/Deinit：
//   - 主机 token/position 缓冲的 packed 填充与补 0；
//   - IoBinder 的输入/输出绑定内容与尺寸；
//   - sample 输出回读、StepResult 填充与 output_slot<0 跳过；
//   - 尺寸/生命周期错误路径与 Deinit 幂等。
//
// 边界：
//   - 不覆盖真实设备、KV Cache、Scheduler 与 Engine 集成；
//   - 不覆盖多输入（>2）的额外输入缓冲行为；
//   - 不覆盖非 int64 token 输入的拒绝路径；
//   - 失败返回非 0，成功打印 ALL TESTS PASSED。
//
// 依赖：
//   Graph 静态库、Runtime/backend 静态库（MockBackend + OpEnvFactory）、
//   Runtime/executor/executor.h、C++17 标准库。

#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Graph/include/attr.h"
#include "Graph/include/graph.h"
#include "Graph/include/graph_env.h"
#include "Graph/include/op.h"
#include "Graph/include/op_type.h"
#include "Graph/include/shape.h"
#include "Graph/include/storage.h"
#include "Graph/include/tensor.h"
#include "Graph/include/types.h"
#include "Runtime/backend/backend.h"
#include "Runtime/backend/mock/half_utils.h"
#include "Runtime/backend/mock/mock_backend.h"
#include "Runtime/common/data_types.h"
#include "Runtime/common/io_binder.h"
#include "Runtime/common/op_env_factory.h"
#include "Runtime/common/plan.h"
#include "Runtime/common/status.h"
#include "Runtime/executor/executor.h"
#include "Utils/cpp/types.h"

namespace {

using mobilenet::Backend;
using mobilenet::BackendConfig;
using mobilenet::Buffer;
using mobilenet::BufferDesc;
using mobilenet::BufferUsage;
using mobilenet::ErrorCode;
using mobilenet::Status;
using mobilenet::mock::FloatToHalf;
using mobilenet::runtime::CreateDefaultExecutor;
using mobilenet::runtime::CreateDefaultOpEnvFactory;
using mobilenet::runtime::ExecutionPlan;
using mobilenet::runtime::IExecutor;
using mobilenet::runtime::IoBinder;
using mobilenet::runtime::KvCacheView;
using mobilenet::runtime::PlanEntry;
using mobilenet::runtime::RequestPhase;
using mobilenet::runtime::RequestStepResult;
using mobilenet::runtime::StepResult;
using mobilenet::runtime_graph::AttrMap;
using mobilenet::runtime_graph::DeviceType;
using mobilenet::runtime_graph::Graph;
using mobilenet::runtime_graph::GraphEnv;
using mobilenet::runtime_graph::Layout;
using mobilenet::runtime_graph::Op;
using mobilenet::runtime_graph::OpEnv;
using mobilenet::runtime_graph::OpType;
using mobilenet::runtime_graph::Precision;
using mobilenet::runtime_graph::Shape;
using mobilenet::runtime_graph::Storage;
using mobilenet::runtime_graph::Tensor;

// 输出一条断言结果。
void Expect(Bool condition, const std::string& message, I32* failures) {
  if (condition) {
    std::cout << "PASS " << message << "\n";
    return;
  }
  std::cout << "FAIL " << message << "\n";
  ++(*failures);
}

// 断言状态码；失败时附带错误消息。
void CheckStatus(const Status& status, ErrorCode expected,
                 const std::string& message, I32* failures) {
  std::string detail = message;
  if (!status.ok()) {
    detail += " [" + status.message() + "]";
  }
  Expect(status.code() == expected, detail, failures);
}

// 把 I64 序列格式化为日志文本。
std::string JoinI64(const std::vector<I64>& values) {
  std::string text = "[";
  for (Usize i = 0; i < values.size(); ++i) {
    if (i != 0) {
      text += ",";
    }
    text += std::to_string(values[i]);
  }
  text += "]";
  return text;
}

// 断言 I64 序列完全一致；失败时打印实际值与期望值。
void ExpectI64Vector(const std::vector<I64>& actual,
                     const std::vector<I64>& expected,
                     const std::string& message, I32* failures) {
  if (actual == expected) {
    Expect(true, message, failures);
    return;
  }
  Expect(false,
         message + " actual=" + JoinI64(actual) +
             " expected=" + JoinI64(expected),
         failures);
}

// 创建指定维度的 Shape（pitch 与 dims 相同）。
Shape MakeShape(std::initializer_list<I64> dims) {
  Shape shape;
  shape.max_dim = static_cast<I64>(dims.size());
  Usize index = 0;
  for (const I64 dim : dims) {
    shape.dims[index] = dim;
    shape.pitch[index] = dim;
    ++index;
  }
  return shape;
}

// 比较两个 Shape 的逻辑维度。
Bool ShapeEquals(const Shape& lhs, const Shape& rhs) {
  if (lhs.max_dim != rhs.max_dim) {
    return false;
  }
  for (I64 i = 0; i < lhs.max_dim; ++i) {
    if (lhs.dims[static_cast<Usize>(i)] !=
        rhs.dims[static_cast<Usize>(i)]) {
      return false;
    }
  }
  return true;
}

// 构造设备 Storage。
Storage DeviceStorage(Buffer* buffer, const Shape& shape,
                      Precision precision, U64 bytes) {
  Storage storage;
  storage.shape = shape;
  storage.precision = precision;
  storage.layout = Layout::kRowMajor;
  storage.device_type = DeviceType::kDevice;
  storage.offset = 0;
  storage.byte_size = bytes;
  storage.data_ptr = buffer;
  storage.reusable = true;
  return storage;
}

// 创建并初始化 Mock 后端。
Status CreateMockBackend(std::unique_ptr<Backend>* backend) {
  Status status =
      mobilenet::mock::CreateMockBackend(BackendConfig{}, backend);
  if (!status.ok()) {
    return status;
  }
  return (*backend)->Init(BackendConfig{});
}

// 对 Graph 的全部 op 创建并挂载 OpEnv。
Bool CreateOpEnvs(Graph* graph, Backend* backend, GraphEnv* graph_env,
                  I32* failures) {
  auto factory = CreateDefaultOpEnvFactory();
  if (factory == nullptr) {
    Expect(false, "factory created", failures);
    return false;
  }
  Bool ok = true;
  for (Op* op : graph->GetOrderlyOPList()) {
    std::unique_ptr<OpEnv> op_env;
    const Status status = factory->Create(op, backend, graph_env, &op_env);
    if (!status.ok() || op_env == nullptr) {
      Expect(false,
             std::string("create OpEnv for ") + op->Name() + ": " +
                 status.message(),
             failures);
      ok = false;
      continue;
    }
    op->SetOpEnv(std::move(op_env));
  }
  return ok;
}

// 测试用 IoBinder：校验、记录并写入主机 Storage。
//
// 功能：
//   模拟 Engine::SetInput/SetOutput：按期望 Storage 校验名称、设备类别、
//   指针、精度、布局、字节数与形状，成功后写入对应 op 的
//   attr["storage"]，并保存主机 Storage 供测试回读缓冲内容。
//
// 边界：
//   不持有主机缓冲；data_ptr 指向 Executor 内部缓冲，只在 Executor 存活
//   期间有效。
class TestIoBinder final : public IoBinder {
 public:
  TestIoBinder(Graph* graph, std::map<std::string, Storage> expected_inputs,
               std::map<std::string, Storage> expected_outputs)
      : expected_inputs_(std::move(expected_inputs)),
        expected_outputs_(std::move(expected_outputs)) {
    for (Op* op : graph->GetAllInputOp()) {
      input_ops_[op->Name()] = op;
    }
    for (Op* op : graph->GetAllOutputOp()) {
      output_ops_[op->Name()] = op;
    }
  }

  Status SetInput(const std::string& name,
                  const Storage& host_storage) override {
    return Bind(name, host_storage, expected_inputs_, &input_ops_,
                &inputs_, &input_count_);
  }

  Status SetOutput(const std::string& name,
                   const Storage& host_storage) override {
    return Bind(name, host_storage, expected_outputs_, &output_ops_,
                &outputs_, &output_count_);
  }

  I32 input_count() const { return input_count_; }

  I32 output_count() const { return output_count_; }

  // 注入一次绑定失败，用于验证 Executor 的错误传递。
  void set_fail_next(Bool fail_next) { fail_next_ = fail_next; }

  const std::map<std::string, Storage>& inputs() const { return inputs_; }

  const std::map<std::string, Storage>& outputs() const {
    return outputs_;
  }

 private:
  // 校验、写入并记录一次绑定。
  Status Bind(const std::string& name, const Storage& host,
              const std::map<std::string, Storage>& expected,
              std::map<std::string, Op*>* ops,
              std::map<std::string, Storage>* bound, I32* count) {
    if (fail_next_) {
      fail_next_ = false;
      return mobilenet::MakeStatus(ErrorCode::kInvalidArgument,
                                   "injected io bind failure");
    }
    const auto it = expected.find(name);
    if (it == expected.end()) {
      return mobilenet::MakeStatus(ErrorCode::kInvalidArgument,
                                   "unknown io name: " + name);
    }
    const auto op_it = ops->find(name);
    if (op_it == ops->end() || op_it->second == nullptr) {
      return mobilenet::MakeStatus(ErrorCode::kInternal,
                                   "missing io op: " + name);
    }
    const Storage& want = it->second;
    if (host.device_type != DeviceType::kHost) {
      return mobilenet::MakeStatus(
          ErrorCode::kInvalidArgument,
          name + ": host storage device_type must be kHost");
    }
    if (host.data_ptr == nullptr) {
      return mobilenet::MakeStatus(
          ErrorCode::kInvalidArgument,
          name + ": host storage data_ptr is null");
    }
    if (host.precision != want.precision) {
      return mobilenet::MakeStatus(
          ErrorCode::kInvalidArgument,
          name + ": host storage precision mismatch");
    }
    if (host.layout != want.layout) {
      return mobilenet::MakeStatus(
          ErrorCode::kInvalidArgument,
          name + ": host storage layout mismatch");
    }
    if (host.byte_size != want.byte_size) {
      return mobilenet::MakeStatus(
          ErrorCode::kInvalidArgument,
          name + ": host storage byte_size mismatch");
    }
    if (!ShapeEquals(host.shape, want.shape)) {
      return mobilenet::MakeStatus(
          ErrorCode::kInvalidArgument,
          name + ": host storage shape mismatch");
    }
    const Status status = op_it->second->SetAttr("storage", host);
    if (!status.ok()) {
      return status;
    }
    (*bound)[name] = host;
    ++(*count);
    return Status::Ok();
  }

  std::map<std::string, Storage> expected_inputs_;
  std::map<std::string, Storage> expected_outputs_;
  std::map<std::string, Op*> input_ops_;
  std::map<std::string, Op*> output_ops_;
  std::map<std::string, Storage> inputs_;
  std::map<std::string, Storage> outputs_;
  I32 input_count_ = 0;
  I32 output_count_ = 0;
  Bool fail_next_ = false;
};

// 从主机 Storage 读取 count 个 I64。
std::vector<I64> ReadBoundI64(const Storage& storage, Usize count) {
  std::vector<I64> values(count, 0);
  if (storage.data_ptr != nullptr && count > 0) {
    std::memcpy(values.data(), storage.data_ptr, count * sizeof(I64));
  }
  return values;
}

// 小图夹具：Input(s) → Gather → Linear → Sample → Output。
//
// 功能：
//   用固定的小常量权重构造可预测的 argmax：
//   embedding[i] = [i, 1]，linear.weight[v] = [2v, -v^2]，
//   因此 logits[v] = 2vi - v^2 = -(v-i)^2 + i^2，最大值在 v=i，
//   sample 结果等于输入 token id。
//
// 边界：
//   - 所有 Buffer 由夹具持有，生命周期覆盖 Graph 与 OpEnv；
//   - 构造失败时 status() 返回首个错误，调用方应跳过后续步骤。
class GraphFixture {
 public:
  GraphFixture(Backend* backend, Bool with_positions)
      : graph_("executor_test_fixture"),
        backend_(backend),
        with_positions_(with_positions) {
    Build();
  }

  // 返回构造状态。
  const Status& status() const { return status_; }

  // 返回图。
  Graph& graph() { return graph_; }

  // 返回图级共享环境。
  GraphEnv& graph_env() { return graph_env_; }

  // 返回 Input op 的期望 Storage 映射。
  std::map<std::string, Storage> ExpectedInputs() const {
    std::map<std::string, Storage> inputs;
    inputs["input.input_ids"] = ids_storage_;
    if (with_positions_) {
      inputs["input.position_ids"] = positions_storage_;
    }
    return inputs;
  }

  // 返回 Output op 的期望 Storage 映射。
  std::map<std::string, Storage> ExpectedOutputs() const {
    std::map<std::string, Storage> outputs;
    outputs["output.tokens"] = sample_storage_;
    return outputs;
  }

 private:
  // 记录首个失败状态。
  void Record(const Status& status, const std::string& step) {
    if (status_.ok()) {
      status_ = mobilenet::PrefixStatus(status, step);
    }
  }

  // 校验状态；失败时记录并返回 false。
  Bool Ok(const Status& status, const std::string& step) {
    if (status.ok()) {
      return true;
    }
    Record(status, step);
    return false;
  }

  // 通过 Backend 创建 Buffer，并由夹具持有。
  Buffer* NewBuffer(U64 bytes) {
    BufferDesc desc;
    desc.bytes = bytes;
    desc.alignment = 64;
    desc.usage = BufferUsage::kIo;
    std::unique_ptr<Buffer> buffer;
    const Status status = backend_->CreateBuffer(desc, &buffer);
    if (!status.ok() || buffer == nullptr) {
      Record(status.ok()
                 ? mobilenet::MakeStatus(ErrorCode::kInternal,
                                         "CreateBuffer returned null")
                 : status,
             "CreateBuffer");
      return nullptr;
    }
    Buffer* raw = buffer.get();
    buffers_.push_back(std::move(buffer));
    return raw;
  }

  // 写入 embedding 权重：第 i 行 [i, 1]。
  static void WriteEmbedding(Buffer* buffer) {
    auto* data = static_cast<FP16*>(buffer->DevicePtr());
    for (I32 i = 0; i < 8; ++i) {
      data[i * 2] = FloatToHalf(static_cast<FP32>(i));
      data[i * 2 + 1] = FloatToHalf(1.0f);
    }
  }

  // 写入 linear 权重：第 v 行 [2v, -v^2]。
  static void WriteLinearWeight(Buffer* buffer) {
    auto* data = static_cast<FP16*>(buffer->DevicePtr());
    for (I32 v = 0; v < 8; ++v) {
      data[v * 2] = FloatToHalf(static_cast<FP32>(2 * v));
      data[v * 2 + 1] = FloatToHalf(-static_cast<FP32>(v * v));
    }
  }

  // 构造图并绑定全部 Storage。
  void Build() {
    const Shape token_shape = MakeShape({1, 4});
    constexpr Usize kTokenCount = 4;
    const U64 token_bytes = kTokenCount * sizeof(I64);

    Tensor* ids = nullptr;
    if (!Ok(graph_.CreateInput("input.input_ids", Precision::kInt64,
                               Layout::kRowMajor, &ids),
            "create input.input_ids")) {
      return;
    }
    Buffer* ids_buffer = NewBuffer(token_bytes);
    if (ids_buffer == nullptr) {
      return;
    }
    ids_storage_ = DeviceStorage(ids_buffer, token_shape,
                                 Precision::kInt64, token_bytes);
    ids->GetStorage() = ids_storage_;

    if (with_positions_) {
      Tensor* positions = nullptr;
      if (!Ok(graph_.CreateInput("input.position_ids",
                                 Precision::kInt64, Layout::kRowMajor,
                                 &positions),
              "create input.position_ids")) {
        return;
      }
      Buffer* positions_buffer = NewBuffer(token_bytes);
      if (positions_buffer == nullptr) {
        return;
      }
      positions_storage_ =
          DeviceStorage(positions_buffer, token_shape,
                        Precision::kInt64, token_bytes);
      positions->GetStorage() = positions_storage_;
    }

    constexpr U64 kWeightBytes = 8 * 2 * sizeof(FP16);
    Buffer* embedding_buffer = NewBuffer(kWeightBytes);
    if (embedding_buffer == nullptr) {
      return;
    }
    WriteEmbedding(embedding_buffer);
    Tensor* embedding = nullptr;
    if (!Ok(graph_.CreateConstant(
                "embedding.weight",
                DeviceStorage(embedding_buffer, MakeShape({8, 2}),
                              Precision::kFloat16, kWeightBytes),
                &embedding),
            "create embedding.weight")) {
      return;
    }

    std::vector<Tensor*> gather_outputs;
    AttrMap gather_attrs;
    gather_attrs["axis"] = I64{0};
    if (!Ok(graph_.CreateOp("gather", {ids, embedding}, 1,
                            OpType::kGather, gather_attrs,
                            &gather_outputs),
            "create gather")) {
      return;
    }
    if (gather_outputs.size() != 1) {
      Record(mobilenet::MakeStatus(ErrorCode::kInternal,
                                   "gather has no output tensor"),
             "build graph");
      return;
    }
    constexpr U64 kGatherBytes = 1 * 4 * 2 * sizeof(FP16);
    Buffer* gather_buffer = NewBuffer(kGatherBytes);
    if (gather_buffer == nullptr) {
      return;
    }
    gather_outputs[0]->GetStorage() =
        DeviceStorage(gather_buffer, MakeShape({1, 4, 2}),
                      Precision::kFloat16, kGatherBytes);

    Buffer* linear_weight_buffer = NewBuffer(kWeightBytes);
    if (linear_weight_buffer == nullptr) {
      return;
    }
    WriteLinearWeight(linear_weight_buffer);
    Tensor* linear_weight = nullptr;
    if (!Ok(graph_.CreateConstant(
                "linear.weight",
                DeviceStorage(linear_weight_buffer, MakeShape({8, 2}),
                              Precision::kFloat16, kWeightBytes),
                &linear_weight),
            "create linear.weight")) {
      return;
    }

    AttrMap linear_attrs;
    linear_attrs["in_features"] = I64{2};
    linear_attrs["out_features"] = I64{8};
    linear_attrs["transpose_weight"] = Bool{true};
    std::vector<Tensor*> linear_outputs;
    if (!Ok(graph_.CreateOp("linear",
                            {gather_outputs[0], linear_weight}, 1,
                            OpType::kLinear, linear_attrs,
                            &linear_outputs),
            "create linear")) {
      return;
    }
    if (linear_outputs.size() != 1) {
      Record(mobilenet::MakeStatus(ErrorCode::kInternal,
                                   "linear has no output tensor"),
             "build graph");
      return;
    }
    constexpr U64 kLogitsBytes = 1 * 4 * 8 * sizeof(FP16);
    Buffer* logits_buffer = NewBuffer(kLogitsBytes);
    if (logits_buffer == nullptr) {
      return;
    }
    linear_outputs[0]->GetStorage() =
        DeviceStorage(logits_buffer, MakeShape({1, 4, 8}),
                      Precision::kFloat16, kLogitsBytes);

    Tensor* sample_out = nullptr;
    if (!Ok(graph_.CreateSample("sample", linear_outputs[0],
                                &sample_out),
            "create sample")) {
      return;
    }
    Buffer* sample_buffer = NewBuffer(token_bytes);
    if (sample_buffer == nullptr) {
      return;
    }
    sample_storage_ = DeviceStorage(sample_buffer, token_shape,
                                    Precision::kInt64, token_bytes);
    sample_out->GetStorage() = sample_storage_;

    if (!Ok(graph_.CreateOutput("output.tokens", sample_out),
            "create output.tokens")) {
      return;
    }
    Ok(graph_.SortOp(), "sort graph");
  }

  GraphEnv graph_env_;
  Graph graph_;
  Backend* backend_ = nullptr;
  Bool with_positions_ = false;
  Status status_;
  std::vector<std::unique_ptr<Buffer>> buffers_;
  Storage ids_storage_;
  Storage positions_storage_;
  Storage sample_storage_;
};

// 构造一个 PlanEntry。
PlanEntry MakeEntry(U64 request_id, RequestPhase phase,
                    I32 packed_offset, std::vector<I64> token_ids,
                    std::vector<I64> positions, I32 output_slot) {
  PlanEntry entry;
  entry.request_id = request_id;
  entry.phase = phase;
  entry.packed_offset = packed_offset;
  entry.token_count = static_cast<I32>(token_ids.size());
  entry.token_ids = std::move(token_ids);
  entry.positions = std::move(positions);
  entry.output_slot = output_slot;
  return entry;
}

// 构造单 entry 计划。
ExecutionPlan MakePlan(U64 step_id, PlanEntry entry) {
  ExecutionPlan plan;
  plan.step_id = step_id;
  plan.total_tokens = entry.token_count;
  plan.entries.push_back(std::move(entry));
  return plan;
}

// 验证 Init/Deinit 的错误路径与后端释放。
void TestInitErrorPaths(I32* failures) {
  auto executor = CreateDefaultExecutor();
  Expect(executor != nullptr, "CreateDefaultExecutor returns instance",
         failures);
  if (executor == nullptr) {
    return;
  }
  Expect(executor->backend() == nullptr, "backend is null before Init",
         failures);
  CheckStatus(executor->Deinit(), ErrorCode::kOk,
              "Deinit before Init is idempotent", failures);

  {
    Graph graph("empty");
    TestIoBinder binder(&graph, {}, {});
    CheckStatus(executor->Init(nullptr, &graph, &binder),
                ErrorCode::kInvalidArgument,
                "Init rejects null backend", failures);
  }
  {
    std::unique_ptr<Backend> backend;
    if (!CreateMockBackend(&backend).ok()) {
      Expect(false, "init error mock backend", failures);
      return;
    }
    Graph graph("empty");
    TestIoBinder binder(&graph, {}, {});
    CheckStatus(executor->Init(std::move(backend), &graph, &binder),
                ErrorCode::kUnsupportedOp,
                "Init rejects graph without Input op", failures);
    Expect(executor->backend() == nullptr,
           "failed Init releases backend", failures);
  }
  {
    std::unique_ptr<Backend> backend;
    if (!CreateMockBackend(&backend).ok()) {
      Expect(false, "no-sample mock backend", failures);
      return;
    }
    Graph graph("no_sample");
    Tensor* input = nullptr;
    graph.CreateInput("input.x", Precision::kInt64, Layout::kRowMajor,
                      &input);
    graph.CreateOutput("output.x", input);
    TestIoBinder binder(&graph, {}, {});
    CheckStatus(executor->Init(std::move(backend), &graph, &binder),
                ErrorCode::kUnsupportedOp,
                "Init rejects graph without Sample op", failures);
  }
  {
    std::unique_ptr<Backend> backend;
    if (!CreateMockBackend(&backend).ok()) {
      Expect(false, "null graph mock backend", failures);
      return;
    }
    CheckStatus(executor->Init(std::move(backend), nullptr, nullptr),
                ErrorCode::kInvalidArgument,
                "Init rejects null graph/io_binder", failures);
  }
}

// 验证双输入图的完整执行链。
void TestTwoInputPipeline(I32* failures) {
  std::unique_ptr<Backend> backend;
  if (!CreateMockBackend(&backend).ok()) {
    Expect(false, "two-input mock backend", failures);
    return;
  }
  GraphFixture fixture(backend.get(), true);
  Expect(fixture.status().ok(),
         "build two-input graph: " +
             (fixture.status().ok() ? std::string("ok")
                                    : fixture.status().message()),
         failures);
  if (!fixture.status().ok()) {
    return;
  }
  TestIoBinder binder(&fixture.graph(), fixture.ExpectedInputs(),
                      fixture.ExpectedOutputs());
  Expect(CreateOpEnvs(&fixture.graph(), backend.get(),
                      &fixture.graph_env(), failures),
         "create two-input OpEnvs", failures);

  auto executor = CreateDefaultExecutor();
  Backend* raw_backend = backend.get();
  CheckStatus(executor->Init(std::move(backend), &fixture.graph(),
                             &binder),
              ErrorCode::kOk, "two-input Init succeeds", failures);
  Expect(executor->backend() == raw_backend,
         "backend() returns owned backend", failures);

  // 重复 Init 必须被拒绝，且不影响已接管的 Backend。
  std::unique_ptr<Backend> second_backend;
  if (CreateMockBackend(&second_backend).ok()) {
    CheckStatus(executor->Init(std::move(second_backend),
                               &fixture.graph(), &binder),
                ErrorCode::kStateError, "double Init is rejected",
                failures);
    Expect(executor->backend() == raw_backend,
           "double Init keeps original backend", failures);
  } else {
    Expect(false, "second mock backend created", failures);
  }

  KvCacheView kv_view;

  // 用例 1：单请求 Prefill，4 个 packed token，回读最后一个位置。
  ExecutionPlan prefill = MakePlan(
      1, MakeEntry(7, RequestPhase::kPrefill, 0, {3, 5, 0, 7},
                   {0, 1, 2, 3}, 3));
  StepResult prefill_result;
  CheckStatus(executor->Execute(prefill, kv_view, &prefill_result),
              ErrorCode::kOk, "Execute prefill succeeds", failures);
  Expect(prefill_result.requests.size() == 1,
         "prefill returns one result", failures);
  if (prefill_result.requests.size() == 1) {
    const RequestStepResult& request_result =
        prefill_result.requests[0];
    Expect(request_result.request_id == 7 && request_result.ok &&
               request_result.error == ErrorCode::kOk,
           "prefill result is ok", failures);
    Expect(request_result.token.token_id == 7,
           "prefill token equals argmax of last packed row", failures);
    Expect(request_result.token.select &&
               request_result.token.token_prob == 0.0f &&
               request_result.token.log_prob == 0.0f,
           "prefill token is selected with zero probs", failures);
  }
  Expect(binder.input_count() == 2 && binder.output_count() == 1,
         "binds 2 inputs and 1 output", failures);
  const auto ids_it = binder.inputs().find("input.input_ids");
  const auto positions_it =
      binder.inputs().find("input.position_ids");
  Expect(ids_it != binder.inputs().end() &&
             positions_it != binder.inputs().end(),
         "input names are bound by op name", failures);
  if (ids_it != binder.inputs().end() &&
      positions_it != binder.inputs().end()) {
    ExpectI64Vector(ReadBoundI64(ids_it->second, 4), {3, 5, 0, 7},
                    "token host buffer is packed in plan order",
                    failures);
    ExpectI64Vector(ReadBoundI64(positions_it->second, 4),
                    {0, 1, 2, 3},
                    "position host buffer is packed in plan order",
                    failures);
  }
  const auto output_it = binder.outputs().find("output.tokens");
  Expect(output_it != binder.outputs().end(),
         "output name is bound by op name", failures);
  if (output_it != binder.outputs().end()) {
    ExpectI64Vector(ReadBoundI64(output_it->second, 4),
                    {3, 5, 0, 7},
                    "output host buffer receives all sample rows",
                    failures);
  }

  // 用例 1b：IoBinder 绑定失败必须中止本步并返回 kInvalidArgument。
  binder.set_fail_next(true);
  StepResult bind_failure_result;
  CheckStatus(executor->Execute(prefill, kv_view, &bind_failure_result),
              ErrorCode::kInvalidArgument,
              "Execute propagates IoBinder failure", failures);
  Expect(bind_failure_result.requests.empty(),
         "failed binding leaves StepResult empty", failures);

  // 用例 2：两个 Decode entry，每个 1 token；验证 packed 偏移、补 0 与多结果。
  ExecutionPlan decode;
  decode.step_id = 2;
  decode.total_tokens = 2;
  decode.entries.push_back(MakeEntry(11, RequestPhase::kDecode, 0, {2},
                                     {4}, 0));
  decode.entries.push_back(MakeEntry(12, RequestPhase::kDecode, 1, {6},
                                     {5}, 1));
  StepResult decode_result;
  CheckStatus(executor->Execute(decode, kv_view, &decode_result),
              ErrorCode::kOk, "Execute decode succeeds", failures);
  Expect(decode_result.requests.size() == 2,
         "decode returns two results", failures);
  if (decode_result.requests.size() == 2) {
    Expect(decode_result.requests[0].request_id == 11 &&
               decode_result.requests[0].token.token_id == 2,
           "decode result 0 maps to packed slot 0", failures);
    Expect(decode_result.requests[1].request_id == 12 &&
               decode_result.requests[1].token.token_id == 6,
           "decode result 1 maps to packed slot 1", failures);
  }
  const auto decode_ids_it = binder.inputs().find("input.input_ids");
  const auto decode_positions_it =
      binder.inputs().find("input.position_ids");
  if (decode_ids_it != binder.inputs().end() &&
      decode_positions_it != binder.inputs().end()) {
    ExpectI64Vector(ReadBoundI64(decode_ids_it->second, 4),
                    {2, 6, 0, 0},
                    "decode token buffer pads remainder with zero",
                    failures);
    ExpectI64Vector(ReadBoundI64(decode_positions_it->second, 4),
                    {4, 5, 0, 0},
                    "decode position buffer pads remainder with zero",
                    failures);
  }

  // 用例 3：output_slot<0（未完成 chunked prefill）不产生请求结果。
  ExecutionPlan chunk = MakePlan(
      3, MakeEntry(21, RequestPhase::kPrefill, 0, {1, 2, 3, 4},
                   {0, 1, 2, 3}, -1));
  StepResult chunk_result;
  CheckStatus(executor->Execute(chunk, kv_view, &chunk_result),
              ErrorCode::kOk, "Execute chunked prefill succeeds",
              failures);
  Expect(chunk_result.requests.empty(),
         "output_slot<0 produces no request result", failures);

  // 用例 4：total 超过主机缓冲容量。
  ExecutionPlan oversize = MakePlan(
      4, MakeEntry(31, RequestPhase::kPrefill, 0, {0, 1, 2, 3, 4},
                   {0, 1, 2, 3, 4}, 0));
  StepResult oversize_result;
  CheckStatus(executor->Execute(oversize, kv_view, &oversize_result),
              ErrorCode::kInvalidArgument,
              "Execute rejects oversize packed tokens", failures);

  // 用例 5：plan.total_tokens 与 entry 之和不一致。
  ExecutionPlan mismatch = prefill;
  mismatch.total_tokens = 3;
  StepResult mismatch_result;
  CheckStatus(executor->Execute(mismatch, kv_view, &mismatch_result),
              ErrorCode::kInvalidArgument,
              "Execute rejects total_tokens mismatch", failures);

  // 用例 6：output_slot 越界。
  ExecutionPlan bad_slot = MakePlan(
      5, MakeEntry(41, RequestPhase::kDecode, 0, {1}, {0}, 4));
  StepResult bad_slot_result;
  CheckStatus(executor->Execute(bad_slot, kv_view, &bad_slot_result),
              ErrorCode::kInvalidArgument,
              "Execute rejects out-of-range output_slot", failures);

  // 用例 7：空计划与空 result。
  ExecutionPlan empty_plan;
  StepResult empty_result;
  CheckStatus(executor->Execute(empty_plan, kv_view, &empty_result),
              ErrorCode::kInvalidArgument,
              "Execute rejects empty plan", failures);
  CheckStatus(executor->Execute(prefill, kv_view, nullptr),
              ErrorCode::kInvalidArgument,
              "Execute rejects null result", failures);

  // 用例 8：缺失 OpEnv。
  for (Op* op : fixture.graph().GetOrderlyOPList()) {
    if (op->Name() == "linear") {
      op->ClearOpEnv();
    }
  }
  StepResult missing_env_result;
  CheckStatus(executor->Execute(prefill, kv_view, &missing_env_result),
              ErrorCode::kStateError,
              "Execute rejects missing OpEnv", failures);

  // 清理顺序必须与 Engine Destroy 一致：先清 OpEnv，再 Deinit Executor。
  for (Op* op : fixture.graph().GetOrderlyOPList()) {
    op->ClearOpEnv();
  }
  CheckStatus(executor->Deinit(), ErrorCode::kOk,
              "Deinit releases backend", failures);
  Expect(executor->backend() == nullptr,
         "backend is null after Deinit", failures);
  CheckStatus(executor->Deinit(), ErrorCode::kOk,
              "Deinit is idempotent", failures);
}

// 验证单输入图只填充 token 缓冲。
void TestSingleInputPipeline(I32* failures) {
  std::unique_ptr<Backend> backend;
  if (!CreateMockBackend(&backend).ok()) {
    Expect(false, "single-input mock backend", failures);
    return;
  }
  GraphFixture fixture(backend.get(), false);
  Expect(fixture.status().ok(),
         "build single-input graph: " +
             (fixture.status().ok() ? std::string("ok")
                                    : fixture.status().message()),
         failures);
  if (!fixture.status().ok()) {
    return;
  }
  TestIoBinder binder(&fixture.graph(), fixture.ExpectedInputs(),
                      fixture.ExpectedOutputs());
  Expect(fixture.ExpectedInputs().size() == 1,
         "single-input graph has one Input op", failures);
  Expect(CreateOpEnvs(&fixture.graph(), backend.get(),
                      &fixture.graph_env(), failures),
         "create single-input OpEnvs", failures);

  auto executor = CreateDefaultExecutor();
  CheckStatus(executor->Init(std::move(backend), &fixture.graph(),
                             &binder),
              ErrorCode::kOk, "single-input Init succeeds", failures);

  ExecutionPlan decode;
  decode.step_id = 1;
  decode.total_tokens = 2;
  decode.entries.push_back(MakeEntry(51, RequestPhase::kDecode, 0, {1},
                                     {0}, 0));
  decode.entries.push_back(MakeEntry(52, RequestPhase::kDecode, 1, {4},
                                     {1}, 1));
  StepResult result;
  KvCacheView kv_view;
  CheckStatus(executor->Execute(decode, kv_view, &result),
              ErrorCode::kOk, "single-input Execute succeeds",
              failures);
  Expect(result.requests.size() == 2,
         "single-input returns two results", failures);
  if (result.requests.size() == 2) {
    Expect(result.requests[0].token.token_id == 1 &&
               result.requests[1].token.token_id == 4,
           "single-input tokens equal plan token ids", failures);
  }
  Expect(binder.input_count() == 1,
         "single-input binds exactly one input", failures);
  const auto ids_it = binder.inputs().find("input.input_ids");
  if (ids_it != binder.inputs().end()) {
    ExpectI64Vector(ReadBoundI64(ids_it->second, 4), {1, 4, 0, 0},
                    "single-input token buffer is padded with zero",
                    failures);
  }

  for (Op* op : fixture.graph().GetOrderlyOPList()) {
    op->ClearOpEnv();
  }
  CheckStatus(executor->Deinit(), ErrorCode::kOk,
              "single-input Deinit succeeds", failures);
}

}  // namespace

// 测试入口：运行 Executor 全部用例并返回失败数。
I32 main() {
  I32 failures = 0;
  TestInitErrorPaths(&failures);
  TestTwoInputPipeline(&failures);
  TestSingleInputPipeline(&failures);

  if (failures == 0) {
    std::cout << "ALL TESTS PASSED\n";
    return 0;
  }
  std::cout << "FAILURES: " << failures << "\n";
  return 1;
}
