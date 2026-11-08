// 模块：RuntimeGraph 单元测试
//
// 职责：
//   使用 Mock Backend/Kernel 验证 RuntimeGraph 的 V17 执行前校验、
//   Forward 执行顺序、H2D/D2H、错误透传与 Release 幂等性。
//
// 边界：
//   - 不依赖 LibTorch、GoogleTest 或真实设备；使用最小断言宏；
//   - 不读取模型权重，只构造合成小图；
//   - 测试失败以非零退出码表示。
//
// 依赖：
//   graph/runtime_graph.h、graph/runtime_graph_builder.h、C++ 标准库。

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "graph/runtime_graph.h"
#include "graph/runtime_graph_builder.h"

namespace {

using mobilenet::Backend;
using mobilenet::CopyKind;
using mobilenet::DataType;
using mobilenet::DeviceType;
using mobilenet::ErrorCode;
using mobilenet::Kernel;
using mobilenet::KernelDesc;
using mobilenet::Layout;
using mobilenet::OpType;
using mobilenet::Status;
using mobilenet::Stream;
using mobilenet::graph::ArenaBinding;
using mobilenet::graph::Dim;
using mobilenet::graph::ForwardContext;
using mobilenet::graph::FusedOp;
using mobilenet::graph::IoBinding;
using mobilenet::graph::KernelContext;
using mobilenet::graph::RuntimeEdge;
using mobilenet::graph::RuntimeGraph;
using mobilenet::graph::RuntimeGraphBuilder;
using mobilenet::graph::RuntimeModelConfig;
using mobilenet::graph::RuntimeNode;
using mobilenet::graph::TensorDesc;

// 全局失败计数。
int g_failures = 0;

// 断言宏：失败时打印文件行与表达式，并继续执行后续用例。
#define CHECK_TRUE(condition)                                              \
  do {                                                                     \
    if (!(condition)) {                                                    \
      std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, \
                   #condition);                                            \
      ++g_failures;                                                        \
    }                                                                      \
  } while (false)

#define CHECK_OK(expression) CHECK_TRUE((expression).ok())

#define CHECK_CODE(expression, expected) \
  CHECK_TRUE((expression).code() == (expected))

// 运行顺序记录。
struct Trace {
  std::vector<std::string> events;
  std::vector<std::string> destroyed;
};

// Mock Kernel：把输入字节复制到输出，可配置为返回错误。
class MockKernel : public Kernel {
 public:
  MockKernel(std::string name, Trace* trace, bool fail)
      : name_(std::move(name)), trace_(trace), fail_(fail) {}

  Status Execute(const KernelContext& context) override {
    trace_->events.push_back("execute:" + name_);
    if (fail_) {
      return Status::Error(ErrorCode::kKernelError, "mock kernel failure");
    }
    if (context.num_inputs != 1 || context.num_outputs != 1 ||
        context.input_ptrs == nullptr || context.output_ptrs == nullptr) {
      return Status::Error(ErrorCode::kInvalidArgument,
                           "unexpected kernel context");
    }
    const uint64_t bytes = context.input_tensors[0]->size_bytes;
    std::memcpy(context.output_ptrs[0], context.input_ptrs[0], bytes);
    return Status::Ok();
  }

 private:
  std::string name_;
  Trace* trace_;
  bool fail_;
};

// Mock Backend：实现 H2D/D2H、Kernel 创建销毁与同步，并记录调用顺序。
class MockBackend : public Backend {
 public:
  // 创建后端；trace 非空时记录调用顺序。
  explicit MockBackend(Trace* trace) : trace_(trace) {}

  // 设置第 N 次 DestroyKernel 调用失败；0 表示不注入失败。
  void set_destroy_failure(int32_t call_index) {
    destroy_failure_index_ = call_index;
  }

  // 设置创建出的 Kernel 是否执行失败。
  void set_kernel_failure(bool fail) { kernel_failure_ = fail; }

  Status Upload(void* device_ptr, const void* host_ptr, uint64_t bytes,
                Stream* /*stream*/) override {
    trace_->events.push_back("upload");
    std::memcpy(device_ptr, host_ptr, bytes);
    return Status::Ok();
  }

  Status Download(void* host_ptr, const void* device_ptr, uint64_t bytes,
                  Stream* /*stream*/) override {
    trace_->events.push_back("download");
    std::memcpy(host_ptr, device_ptr, bytes);
    return Status::Ok();
  }

  Status CreateKernel(const KernelDesc& desc, Kernel** kernel) override {
    trace_->events.push_back("create:" + desc.kernel_key);
    *kernel = new MockKernel(desc.kernel_key, trace_, kernel_failure_);
    ++live_kernels_;
    return Status::Ok();
  }

  Status DestroyKernel(Kernel* kernel) override {
    ++destroy_calls_;
    if (destroy_failure_index_ != 0 &&
        destroy_calls_ == destroy_failure_index_) {
      return Status::Error(ErrorCode::kKernelError, "mock destroy failure");
    }
    if (kernel != nullptr) {
      trace_->destroyed.push_back("destroy");
      delete kernel;
      --live_kernels_;
    }
    return Status::Ok();
  }

  Status Synchronize(Stream* /*stream*/) override {
    trace_->events.push_back("synchronize");
    return Status::Ok();
  }

  int32_t live_kernels() const { return live_kernels_; }

 private:
  Trace* trace_;
  int32_t destroy_failure_index_ = 0;
  int32_t destroy_calls_ = 0;
  int32_t live_kernels_ = 0;
  bool kernel_failure_ = false;
};

// 构造 [1, 4] int64 设备张量描述。
TensorDesc MakeInt64Tensor(uint64_t offset) {
  TensorDesc tensor;
  tensor.dtype = DataType::kInt64;
  tensor.shape = {Dim::Constant(1), Dim::Constant(4)};
  tensor.stride = {4, 1};
  tensor.layout = Layout::kRowMajor;
  tensor.device = DeviceType::kDevice;
  tensor.offset = offset;
  tensor.size_bytes = 4 * sizeof(int64_t);
  return tensor;
}

// 构造 [1, 4] int64 主机张量描述。
TensorDesc MakeHostTensor() {
  TensorDesc tensor = MakeInt64Tensor(0);
  tensor.device = DeviceType::kHost;
  return tensor;
}

// 构造最小模型配置。
RuntimeModelConfig MakeModelConfig() {
  RuntimeModelConfig config;
  config.model_family = "test";
  config.num_layers = 1;
  config.hidden_size = 4;
  config.intermediate_size = 8;
  config.num_attention_heads = 1;
  config.num_kv_heads = 1;
  config.head_dim = 4;
  config.vocab_size = 8;
  config.max_context_length = 8;
  config.norm_eps = 1e-6;
  config.activation = "silu";
  config.dtype = DataType::kInt64;
  config.rope.theta = 10000.0;
  config.rope.layout = mobilenet::RopeLayout::kNeox;
  config.rope.scaling_factor = 1.0;
  return config;
}

// 构造一个计算节点。
RuntimeNode MakeCastNode(std::string name, mobilenet::graph::EdgeId input,
                         mobilenet::graph::EdgeId output) {
  RuntimeNode node;
  node.name = std::move(name);
  node.op_type = OpType::kCast;
  FusedOp fused_op;
  fused_op.source_name = "test.cast";
  fused_op.op_kind = "cast";
  node.fused_ops.push_back(std::move(fused_op));
  node.input_edges = {input};
  node.output_edges = {output};
  node.precision = DataType::kInt64;
  node.kernel_key = "cast";
  return node;
}

// 构造测试图。
//
// 参数：
//   compute_nodes：计算节点数量，取 1 或 2；
//   out_offset：输出边在 arena 中的字节偏移；
//   graph：输出图指针。
// 返回：
//   构图状态。
Status BuildTestGraph(int32_t compute_nodes, uint64_t out_offset,
                      std::unique_ptr<RuntimeGraph>* graph) {
  RuntimeGraphBuilder builder;
  Status status = builder.SetModelConfig(MakeModelConfig());
  if (!status.ok()) {
    return status;
  }

  RuntimeEdge ids_edge;
  ids_edge.name = "ids";
  ids_edge.tensor = MakeInt64Tensor(0);
  ids_edge.arena_id = 0;
  ids_edge.memory_offset = 0;
  mobilenet::graph::EdgeId ids_id = mobilenet::graph::kInvalidEdgeId;
  status = builder.AddEdge(std::move(ids_edge), &ids_id);
  if (!status.ok()) {
    return status;
  }

  mobilenet::graph::EdgeId mid_id = mobilenet::graph::kInvalidEdgeId;
  if (compute_nodes == 2) {
    RuntimeEdge mid_edge;
    mid_edge.name = "mid";
    mid_edge.tensor = MakeInt64Tensor(0);
    mid_edge.arena_id = 0;
    mid_edge.memory_offset = 32;
    status = builder.AddEdge(std::move(mid_edge), &mid_id);
    if (!status.ok()) {
      return status;
    }
  }

  RuntimeEdge out_edge;
  out_edge.name = "out";
  out_edge.tensor = MakeInt64Tensor(0);
  out_edge.arena_id = 0;
  out_edge.memory_offset = out_offset;
  mobilenet::graph::EdgeId out_id = mobilenet::graph::kInvalidEdgeId;
  status = builder.AddEdge(std::move(out_edge), &out_id);
  if (!status.ok()) {
    return status;
  }

  RuntimeNode input;
  input.name = "input.ids";
  input.op_type = OpType::kInput;
  input.output_edges = {ids_id};
  input.precision = DataType::kInt64;
  input.binding_name = "ids";
  input.copy_kind = CopyKind::kH2D;
  input.host_tensor = MakeHostTensor();
  std::vector<mobilenet::graph::NodeId> input_nodes;
  mobilenet::graph::NodeId node_id = mobilenet::graph::kInvalidNodeId;
  status = builder.AddNode(std::move(input), &node_id);
  if (!status.ok()) {
    return status;
  }
  input_nodes.push_back(node_id);

  mobilenet::graph::EdgeId previous = ids_id;
  for (int32_t index = 0; index < compute_nodes; ++index) {
    const mobilenet::graph::EdgeId next =
        index + 1 == compute_nodes ? out_id : mid_id;
    const std::string name = "cast." + std::to_string(index);
    status = builder.AddNode(MakeCastNode(name, previous, next), &node_id);
    if (!status.ok()) {
      return status;
    }
    previous = next;
  }

  RuntimeNode output;
  output.name = "output.ids";
  output.op_type = OpType::kOutput;
  output.input_edges = {out_id};
  output.precision = DataType::kInt64;
  output.binding_name = "ids_out";
  output.copy_kind = CopyKind::kD2H;
  output.host_tensor = MakeHostTensor();
  status = builder.AddNode(std::move(output), &node_id);
  if (!status.ok()) {
    return status;
  }
  std::vector<mobilenet::graph::NodeId> output_nodes = {node_id};

  status = builder.SetInputNodes(std::move(input_nodes));
  if (!status.ok()) {
    return status;
  }
  status = builder.SetOutputNodes(std::move(output_nodes));
  if (!status.ok()) {
    return status;
  }
  return builder.Finalize(graph);
}

// 构造 Forward 上下文；host_inputs/host_outputs 指向调用方缓冲区。
ForwardContext MakeContext(std::vector<int64_t>* input,
                           std::vector<int64_t>* output, MockBackend* backend) {
  ForwardContext context;
  IoBinding input_binding;
  input_binding.host_ptr = input->data();
  input_binding.bytes = input->size() * sizeof(int64_t);
  input_binding.shape = {Dim::Constant(1), Dim::Constant(4)};
  input_binding.dtype = DataType::kInt64;
  input_binding.device = DeviceType::kHost;
  context.host_inputs.emplace("ids", input_binding);

  IoBinding output_binding = input_binding;
  output_binding.host_ptr = output->data();
  context.host_outputs.emplace("ids_out", output_binding);
  context.backend = backend;
  return context;
}

// 用例：Forward 顺序、H2D/D2H 数据与控制流同步。
void TestForwardCopiesData() {
  std::unique_ptr<RuntimeGraph> graph;
  CHECK_OK(BuildTestGraph(1, 32, &graph));
  CHECK_TRUE(graph != nullptr);
  mobilenet::graph::ValidationReport report;
  CHECK_OK(graph->Validate(&report));

  Trace trace;
  MockBackend backend(&trace);
  CHECK_OK(graph->BindBackend(&backend, nullptr));
  std::vector<uint8_t> arena(128, 0);
  CHECK_OK(graph->BindArena(0, arena.data(), arena.size(), nullptr));
  mobilenet::graph::NodeId cast_id = graph->FindNode("cast.0");
  CHECK_TRUE(cast_id != mobilenet::graph::kInvalidNodeId);
  Kernel* kernel = nullptr;
  KernelDesc desc;
  desc.op_type = OpType::kCast;
  desc.precision = DataType::kInt64;
  desc.kernel_key = "cast";
  CHECK_OK(backend.CreateKernel(desc, &kernel));
  CHECK_OK(graph->BindKernel(cast_id, kernel, nullptr));
  trace.events.clear();

  std::vector<int64_t> input = {11, 22, 33, 44};
  std::vector<int64_t> output = {0, 0, 0, 0};
  ForwardContext context = MakeContext(&input, &output, &backend);
  CHECK_OK(graph->ValidateExecution(&context, &report));
  CHECK_OK(graph->Forward(&context, &report));
  CHECK_TRUE(output == input);
  CHECK_TRUE(trace.events.size() == 4);
  if (trace.events.size() == 4) {
    CHECK_TRUE(trace.events[0] == "upload");
    CHECK_TRUE(trace.events[1] == "execute:cast");
    CHECK_TRUE(trace.events[2] == "download");
    CHECK_TRUE(trace.events[3] == "synchronize");
  }
  CHECK_OK(graph->Release(nullptr));
  CHECK_TRUE(backend.live_kernels() == 0);
}

// 用例：缺少 Kernel、缺少 I/O 绑定时 V17 拒绝执行。
void TestExecutionValidationRejectsMissingBindings() {
  std::unique_ptr<RuntimeGraph> graph;
  CHECK_OK(BuildTestGraph(1, 32, &graph));
  Trace trace;
  MockBackend backend(&trace);
  CHECK_OK(graph->BindBackend(&backend, nullptr));
  std::vector<uint8_t> arena(128, 0);
  CHECK_OK(graph->BindArena(0, arena.data(), arena.size(), nullptr));

  std::vector<int64_t> input = {1, 2, 3, 4};
  std::vector<int64_t> output = {0, 0, 0, 0};
  ForwardContext context = MakeContext(&input, &output, &backend);
  mobilenet::graph::ValidationReport report;
  CHECK_CODE(graph->ValidateExecution(&context, &report),
             ErrorCode::kStateError);
  CHECK_TRUE(!report.ok());

  context.host_inputs.clear();
  CHECK_CODE(graph->Forward(&context, &report), ErrorCode::kInvalidArgument);
  CHECK_OK(graph->Release(nullptr));
}

// 用例：重复绑定与非法绑定被拒绝。
void TestBindRules() {
  std::unique_ptr<RuntimeGraph> graph;
  CHECK_OK(BuildTestGraph(1, 32, &graph));
  Trace trace;
  MockBackend backend(&trace);
  CHECK_OK(graph->BindBackend(&backend, nullptr));
  CHECK_CODE(graph->BindBackend(&backend, nullptr), ErrorCode::kStateError);

  std::vector<uint8_t> arena(128, 0);
  CHECK_OK(graph->BindArena(0, arena.data(), arena.size(), nullptr));
  CHECK_CODE(graph->BindArena(0, arena.data(), arena.size(), nullptr),
             ErrorCode::kStateError);
  CHECK_CODE(graph->BindArena(1, nullptr, 16, nullptr),
             ErrorCode::kInvalidArgument);

  mobilenet::graph::NodeId input_id = graph->FindNode("input.ids");
  KernelDesc desc;
  desc.op_type = OpType::kCast;
  desc.precision = DataType::kInt64;
  desc.kernel_key = "cast";
  Kernel* kernel = nullptr;
  CHECK_OK(backend.CreateKernel(desc, &kernel));
  CHECK_CODE(graph->BindKernel(input_id, kernel, nullptr),
             ErrorCode::kInvalidArgument);
  CHECK_OK(backend.DestroyKernel(kernel));
  CHECK_OK(graph->Release(nullptr));
}

// 用例：arena 越界被 V17 拒绝。
void TestArenaRangeValidation() {
  std::unique_ptr<RuntimeGraph> graph;
  CHECK_OK(BuildTestGraph(1, 100, &graph));
  Trace trace;
  MockBackend backend(&trace);
  CHECK_OK(graph->BindBackend(&backend, nullptr));
  mobilenet::graph::NodeId cast_id = graph->FindNode("cast.0");
  Kernel* kernel = nullptr;
  KernelDesc desc;
  desc.op_type = OpType::kCast;
  desc.precision = DataType::kInt64;
  desc.kernel_key = "cast";
  CHECK_OK(backend.CreateKernel(desc, &kernel));
  CHECK_OK(graph->BindKernel(cast_id, kernel, nullptr));

  std::vector<uint8_t> arena(128, 0);
  CHECK_OK(graph->BindArena(0, arena.data(), arena.size(), nullptr));
  std::vector<int64_t> input = {1, 2, 3, 4};
  std::vector<int64_t> output = {0, 0, 0, 0};
  ForwardContext context = MakeContext(&input, &output, &backend);
  mobilenet::graph::ValidationReport report;
  CHECK_CODE(graph->ValidateExecution(&context, &report),
             ErrorCode::kBufferError);
  CHECK_OK(graph->Release(nullptr));
}

// 用例：Kernel 错误透传并附加节点名。
void TestKernelErrorPropagation() {
  std::unique_ptr<RuntimeGraph> graph;
  CHECK_OK(BuildTestGraph(1, 32, &graph));
  Trace trace;
  MockBackend backend(&trace);
  backend.set_kernel_failure(true);
  CHECK_OK(graph->BindBackend(&backend, nullptr));
  std::vector<uint8_t> arena(128, 0);
  CHECK_OK(graph->BindArena(0, arena.data(), arena.size(), nullptr));
  mobilenet::graph::NodeId cast_id = graph->FindNode("cast.0");
  Kernel* kernel = nullptr;
  KernelDesc desc;
  desc.op_type = OpType::kCast;
  desc.precision = DataType::kInt64;
  desc.kernel_key = "cast";
  CHECK_OK(backend.CreateKernel(desc, &kernel));
  CHECK_OK(graph->BindKernel(cast_id, kernel, nullptr));

  std::vector<int64_t> input = {1, 2, 3, 4};
  std::vector<int64_t> output = {0, 0, 0, 0};
  ForwardContext context = MakeContext(&input, &output, &backend);
  const Status status = graph->Forward(&context, nullptr);
  CHECK_CODE(status, ErrorCode::kKernelError);
  CHECK_TRUE(status.message().find("cast.0") != std::string::npos);
  CHECK_OK(graph->Release(nullptr));
}

// 用例：Release 逆拓扑序销毁 Kernel、单个销毁失败不影响其余 Kernel。
void TestReleaseDestroysAllKernels() {
  std::unique_ptr<RuntimeGraph> graph;
  CHECK_OK(BuildTestGraph(2, 64, &graph));
  Trace trace;
  MockBackend backend(&trace);
  backend.set_destroy_failure(1);
  CHECK_OK(graph->BindBackend(&backend, nullptr));
  std::vector<uint8_t> arena(128, 0);
  CHECK_OK(graph->BindArena(0, arena.data(), arena.size(), nullptr));

  std::vector<std::string> names = {"cast.0", "cast.1"};
  for (const std::string& name : names) {
    Kernel* kernel = nullptr;
    KernelDesc desc;
    desc.op_type = OpType::kCast;
    desc.precision = DataType::kInt64;
    desc.kernel_key = name;
    CHECK_OK(backend.CreateKernel(desc, &kernel));
    CHECK_OK(graph->BindKernel(graph->FindNode(name), kernel, nullptr));
  }
  const Status status = graph->Release(nullptr);
  CHECK_CODE(status, ErrorCode::kKernelError);
  CHECK_TRUE(backend.live_kernels() == 1);
  CHECK_OK(graph->Release(nullptr));
  CHECK_TRUE(backend.live_kernels() == 1);
  CHECK_TRUE(graph->released());

  ForwardContext context;
  CHECK_CODE(graph->Forward(&context, nullptr), ErrorCode::kStateError);
}

// 用例：Canonical JSON 多次输出一致。
void TestCanonicalJsonDeterminism() {
  std::unique_ptr<RuntimeGraph> graph;
  CHECK_OK(BuildTestGraph(1, 32, &graph));
  const std::string first = graph->ToCanonicalJson();
  const std::string second = graph->ToCanonicalJson();
  CHECK_TRUE(!first.empty());
  CHECK_TRUE(first == second);
  CHECK_TRUE(first.find("\"nodes\"") != std::string::npos);
  CHECK_OK(graph->Release(nullptr));
}

}  // namespace

// 测试入口：运行全部用例并返回失败数。
int main() {
  TestForwardCopiesData();
  TestExecutionValidationRejectsMissingBindings();
  TestBindRules();
  TestArenaRangeValidation();
  TestKernelErrorPropagation();
  TestReleaseDestroysAllKernels();
  TestCanonicalJsonDeterminism();
  if (g_failures == 0) {
    std::printf("PASS runtime_graph tests\n");
    return 0;
  }
  std::fprintf(stderr, "FAIL runtime_graph tests: %d check(s)\n", g_failures);
  return 1;
}
