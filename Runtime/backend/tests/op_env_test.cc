// 模块：OpEnv/Graph 执行链测试
//
// 职责：
//   用 C++ 构造一个小 Graph（Input → Add → Output），经默认 OpEnvFactory
//   创建 OpEnv，设置 StepContext，依次执行 H2D、Kernel、D2H，并校验结果；
//   另验证未知 op 与缺失 attr 的加载期错误。
//
// 边界：
//   - 不经过 Loader/Serializer（P2/P3）；Buffer 由测试直接创建；
//   - 不覆盖 8 个算子的 torch 精度对比；
//   - 失败返回非 0，成功打印 ALL TESTS PASSED。
//
// 依赖：
//   Graph 静态库、Runtime/backend 公共头、
//   Runtime/common/op_env_factory.h、mock/half_utils.h。

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Graph/include/attr.h"
#include "Graph/include/graph.h"
#include "Graph/include/graph_env.h"
#include "Graph/include/op.h"
#include "Graph/include/shape.h"
#include "Graph/include/storage.h"
#include "Graph/include/tensor.h"
#include "Graph/include/types.h"
#include "Runtime/backend/backend.h"
#include "Runtime/backend/backend_registry.h"
#include "Runtime/backend/exec_op_env.h"
#include "Runtime/backend/kernel_types.h"
#include "Runtime/backend/mock/half_utils.h"
#include "Runtime/backend/ops/op_spec.h"
#include "Runtime/common/data_types.h"
#include "Runtime/common/op_env_factory.h"
#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace {

using mobilenet::Backend;
using mobilenet::BackendConfig;
using mobilenet::BackendRegistry;
using mobilenet::Buffer;
using mobilenet::BufferDesc;
using mobilenet::BufferUsage;
using mobilenet::DataType;
using mobilenet::ErrorCode;
using mobilenet::Status;
using mobilenet::mock::FloatToHalf;
using mobilenet::mock::HalfToFloat;
using mobilenet::runtime::CreateDefaultOpEnvFactory;
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

// 输出浮点比较断言。
void ExpectNear(FP32 actual, FP32 expected, FP32 tolerance,
                const std::string& message, I32* failures) {
  Expect(std::fabs(actual - expected) <= tolerance, message, failures);
}

// 创建指定维度的 Shape。
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

// 创建 Mock 设备缓冲。
Bool CreateBuffer(Backend* backend, U64 bytes,
                  std::unique_ptr<Buffer>* buffer) {
  BufferDesc desc;
  desc.bytes = bytes;
  desc.alignment = 64;
  desc.usage = BufferUsage::kPersistent;
  return backend->CreateBuffer(desc, buffer).ok() && *buffer != nullptr;
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

// 构造主机 Storage。
Storage HostStorage(void* data, const Shape& shape, Precision precision,
                    U64 bytes) {
  Storage storage;
  storage.shape = shape;
  storage.precision = precision;
  storage.layout = Layout::kRowMajor;
  storage.device_type = DeviceType::kHost;
  storage.offset = 0;
  storage.byte_size = bytes;
  storage.data_ptr = data;
  storage.reusable = false;
  return storage;
}

// 把 FP32 序列写成 half 设备张量。
void WriteHalves(Buffer* buffer, const std::vector<FP32>& values) {
  auto* data = static_cast<U16*>(buffer->DevicePtr());
  for (Usize i = 0; i < values.size(); ++i) {
    data[i] = FloatToHalf(values[i]);
  }
}

// 读取 half 设备张量。
std::vector<FP32> ReadHalves(Buffer* buffer, Usize count) {
  const auto* data = static_cast<const U16*>(buffer->DevicePtr());
  std::vector<FP32> values(count, 0.0f);
  for (Usize i = 0; i < count; ++i) {
    values[i] = HalfToFloat(data[i]);
  }
  return values;
}

// 把 FP32 序列写入主机 half 数组。
void WriteHostHalves(U16* data, const std::vector<FP32>& values) {
  for (Usize i = 0; i < values.size(); ++i) {
    data[i] = FloatToHalf(values[i]);
  }
}

// 创建并初始化 Mock 后端。
Status CreateMockBackend(std::unique_ptr<Backend>* backend) {
  Status status = BackendRegistry::Instance().Create(
      "mock", BackendConfig{}, backend);
  if (!status.ok()) {
    return status;
  }
  return (*backend)->Init(BackendConfig{});
}

// 对 Graph 的全部 op 创建 OpEnv。
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
    const Status status =
        factory->Create(op, backend, graph_env, &op_env);
    if (!status.ok() || op_env == nullptr) {
      Expect(false, std::string("create OpEnv for ") + op->Name(),
             failures);
      ok = false;
      continue;
    }
    op->SetOpEnv(std::move(op_env));
  }
  return ok;
}

// 设置 StepContext 并执行全部 op。
Bool ExecuteGraph(Graph* graph, Backend* backend, I32* failures) {
  mobilenet::StepContext step;
  step.backend = backend;
  step.batch = 1;
  step.packed_tokens = 4;
  // 全量 attention 单算子测试不需要 KV 元数据。

  Bool ok = true;
  for (Op* op : graph->GetOrderlyOPList()) {
    OpEnv* op_env = op->GetOpEnv();
    if (op_env == nullptr) {
      Expect(false, std::string("missing OpEnv for ") + op->Name(),
             failures);
      ok = false;
      continue;
    }
    const Status context_status = mobilenet::SetStepContext(op_env, &step);
    if (!context_status.ok()) {
      Expect(false, std::string("set StepContext for ") + op->Name(),
             failures);
      ok = false;
      continue;
    }
    const Status forward_status = op_env->Forward();
    if (!forward_status.ok()) {
      Expect(false, std::string("Forward for ") + op->Name() + ": " +
                        forward_status.message(),
             failures);
      ok = false;
    }
  }
  return ok;
}

// 验证 8 个融合算子 OpSpec 已登记，未支持类型返回空。
void TestSpecRegistry(I32* failures) {
  const OpType supported[] = {
      OpType::kGather,
      OpType::kRmsNorm,
      OpType::kLinear,
      OpType::kLinearSwiGlu,
      OpType::kLinearPerHeadRMSNormRope,
      OpType::kMegaAttention,
      OpType::kAdd,
      OpType::kSample,
  };
  for (const OpType type : supported) {
    Expect(mobilenet::FindOpSpec(type) != nullptr,
           std::string("spec registered for ") +
               mobilenet::runtime_graph::OpTypeToStr(type),
           failures);
  }
  Expect(mobilenet::FindOpSpec(OpType::kRope) == nullptr,
         "unsupported OpType has no spec", failures);
}

// 验证 Input → Add → Output 的完整执行链。
void TestAddGraph(I32* failures) {
  std::unique_ptr<Backend> backend;
  if (!CreateMockBackend(&backend).ok()) {
    Expect(false, "add graph backend init", failures);
    return;
  }

  std::unique_ptr<Buffer> input_device;
  std::unique_ptr<Buffer> rhs_device;
  std::unique_ptr<Buffer> output_device;
  const Shape tensor_shape = MakeShape({1, 2, 2});
  constexpr U64 kTensorBytes = 4 * sizeof(U16);
  if (!CreateBuffer(backend.get(), kTensorBytes, &input_device) ||
      !CreateBuffer(backend.get(), kTensorBytes, &rhs_device) ||
      !CreateBuffer(backend.get(), kTensorBytes, &output_device)) {
    Expect(false, "add graph buffers", failures);
    return;
  }
  WriteHalves(input_device.get(), {1.0f, 2.0f, 3.0f, 4.0f});
  WriteHalves(rhs_device.get(), {10.0f, 20.0f, 30.0f, 40.0f});
  WriteHalves(output_device.get(), {0.0f, 0.0f, 0.0f, 0.0f});

  std::vector<U16> input_host(4, 0);
  WriteHostHalves(input_host.data(), {1.0f, 2.0f, 3.0f, 4.0f});
  std::vector<U16> output_host(4, 0);

  Graph graph("op_env_add");
  Tensor* input = nullptr;
  Expect(graph.CreateInput("input.a", Precision::kFloat16,
                           Layout::kRowMajor, &input)
             .ok(),
         "create input", failures);
  if (input != nullptr) {
    input->GetStorage() = DeviceStorage(input_device.get(), tensor_shape,
                                        Precision::kFloat16,
                                        kTensorBytes);
    Expect(graph.GetAllInputOp()[0]
               ->SetAttr("storage",
                         HostStorage(input_host.data(), tensor_shape,
                                     Precision::kFloat16, kTensorBytes))
               .ok(),
           "bind input host storage", failures);
  }

  Tensor* rhs = nullptr;
  Expect(graph.CreateConstant("rhs",
                              DeviceStorage(rhs_device.get(), tensor_shape,
                                            Precision::kFloat16,
                                            kTensorBytes),
                              &rhs)
             .ok(),
         "create rhs constant", failures);

  AttrMap attrs;
  attrs["precision"] = std::string("float16");
  std::vector<Tensor*> outputs;
  Expect(graph.CreateOp("add", {input, rhs}, 1, OpType::kAdd, attrs,
                        &outputs)
             .ok() &&
             outputs.size() == 1,
         "create add op", failures);
  if (outputs.size() == 1) {
    outputs[0]->GetStorage() =
        DeviceStorage(output_device.get(), tensor_shape,
                      Precision::kFloat16, kTensorBytes);
  }
  Expect(graph.CreateOutput("output.y", outputs.empty() ? nullptr
                                                        : outputs[0])
             .ok(),
         "create output", failures);
  if (!graph.GetAllOutputOp().empty()) {
    Expect(graph.GetAllOutputOp()[0]
               ->SetAttr("storage",
                         HostStorage(output_host.data(), tensor_shape,
                                     Precision::kFloat16, kTensorBytes))
               .ok(),
           "bind output host storage", failures);
  }
  Expect(graph.SortOp().ok(), "sort add graph", failures);

  GraphEnv graph_env;
  Expect(CreateOpEnvs(&graph, backend.get(), &graph_env, failures),
         "create add OpEnvs", failures);
  Expect(ExecuteGraph(&graph, backend.get(), failures),
         "execute add graph", failures);
  Expect(backend->Synchronize(nullptr).ok(), "synchronize add graph",
         failures);

  const std::vector<FP32> values = ReadHalves(output_device.get(), 4);
  ExpectNear(values[0], 11.0f, 1e-3f, "graph add value 0", failures);
  ExpectNear(values[1], 22.0f, 1e-3f, "graph add value 1", failures);
  ExpectNear(values[2], 33.0f, 1e-3f, "graph add value 2", failures);
  ExpectNear(values[3], 44.0f, 1e-3f, "graph add value 3", failures);

  const std::vector<FP32> host_values = {
      HalfToFloat(output_host[0]), HalfToFloat(output_host[1]),
      HalfToFloat(output_host[2]), HalfToFloat(output_host[3])};
  ExpectNear(host_values[0], 11.0f, 1e-3f, "D2H add value 0", failures);
  ExpectNear(host_values[1], 22.0f, 1e-3f, "D2H add value 1", failures);
  ExpectNear(host_values[2], 33.0f, 1e-3f, "D2H add value 2", failures);
  ExpectNear(host_values[3], 44.0f, 1e-3f, "D2H add value 3", failures);

  for (Op* op : graph.GetOrderlyOPList()) {
    op->ClearOpEnv();
  }
  Expect(backend->Deinit().ok(), "add graph Deinit", failures);
}

// 验证未知 op 在工厂创建期返回 kUnsupportedOp。
void TestUnsupportedOp(I32* failures) {
  std::unique_ptr<Backend> backend;
  if (!CreateMockBackend(&backend).ok()) {
    Expect(false, "unsupported backend init", failures);
    return;
  }
  Graph graph("op_env_unsupported");
  Tensor* input = nullptr;
  graph.CreateInput("input.x", Precision::kFloat16, Layout::kRowMajor,
                    &input);
  std::vector<Tensor*> outputs;
  graph.CreateOp("rope", {input}, 1, OpType::kRope, AttrMap{}, &outputs);
  graph.CreateOutput("output.x", outputs.empty() ? nullptr : outputs[0]);
  Expect(graph.SortOp().ok(), "sort unsupported graph", failures);

  auto factory = CreateDefaultOpEnvFactory();
  GraphEnv graph_env;
  for (Op* op : graph.GetOrderlyOPList()) {
    if (op->Type() != OpType::kRope) {
      continue;
    }
    std::unique_ptr<OpEnv> op_env;
    Expect(factory->Create(op, backend.get(), &graph_env, &op_env).code() ==
               ErrorCode::kUnsupportedOp,
           "unsupported op rejected", failures);
  }
  Expect(backend->Deinit().ok(), "unsupported Deinit", failures);
}

// 验证缺失 attr 在加载期返回 kInvalidArgument。
void TestMissingAttr(I32* failures) {
  std::unique_ptr<Backend> backend;
  if (!CreateMockBackend(&backend).ok()) {
    Expect(false, "missing attr backend init", failures);
    return;
  }
  Graph graph("op_env_missing_attr");
  Tensor* input = nullptr;
  graph.CreateInput("input.x", Precision::kFloat16, Layout::kRowMajor,
                    &input);
  std::vector<Tensor*> outputs;
  graph.CreateOp("linear", {input}, 1, OpType::kLinear, AttrMap{},
                 &outputs);
  graph.CreateOutput("output.x", outputs.empty() ? nullptr : outputs[0]);
  Expect(graph.SortOp().ok(), "sort missing attr graph", failures);

  auto factory = CreateDefaultOpEnvFactory();
  GraphEnv graph_env;
  for (Op* op : graph.GetOrderlyOPList()) {
    if (op->Name() != "linear") {
      continue;
    }
    std::unique_ptr<OpEnv> op_env;
    Expect(factory->Create(op, backend.get(), &graph_env, &op_env).code() ==
               ErrorCode::kInvalidArgument,
           "missing attr rejected", failures);
  }
  Expect(backend->Deinit().ok(), "missing attr Deinit", failures);
}

}  // namespace

I32 main() {
  I32 failures = 0;
  TestSpecRegistry(&failures);
  TestAddGraph(&failures);
  TestUnsupportedOp(&failures);
  TestMissingAttr(&failures);

  if (failures == 0) {
    std::cout << "ALL TESTS PASSED\n";
    return 0;
  }
  std::cout << "FAILURES: " << failures << "\n";
  return 1;
}
