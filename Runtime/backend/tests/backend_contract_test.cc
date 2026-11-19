// 模块：Mock 后端契约测试
//
// 职责：
//   验证 Mock 后端的注册、生命周期、Buffer、H2D/D2H、ConvertParam、
//   Kernel 创建/销毁与 8 个 CPU Kernel 的基础数值行为。
//
// 边界：
//   - 不经过 Loader/Graph/OpEnv，只测试 Backend/Kernel 契约；
//   - 不覆盖 torch 精度对比（P2/P3）与 OCL 后端（P4）；
//   - 失败返回非 0，成功打印 ALL TESTS PASSED。
//
// 依赖：
//   Runtime/backend 公共头、mock/half_utils.h、Graph 静态库。

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "Graph/include/shape.h"
#include "Graph/include/types.h"
#include "Runtime/backend/backend.h"
#include "Runtime/backend/backend_registry.h"
#include "Runtime/backend/kernel.h"
#include "Runtime/backend/kernel_context.h"
#include "Runtime/backend/kernel_types.h"
#include "Runtime/backend/mock/half_utils.h"
#include "Runtime/common/data_types.h"
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
using mobilenet::Kernel;
using mobilenet::KernelDesc;
using mobilenet::ParamDesc;
using mobilenet::Status;
using mobilenet::mock::BFloat16ToHalf;
using mobilenet::mock::FloatToHalf;
using mobilenet::mock::HalfToFloat;
using mobilenet::runtime_graph::KernelContext;
using mobilenet::runtime_graph::OpType;
using mobilenet::runtime_graph::Shape;
using mobilenet::runtime_graph::TensorView;

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
void ExpectNear(F32 actual, F32 expected, F32 tolerance,
                const std::string& message, I32* failures) {
  Expect(std::fabs(actual - expected) <= tolerance, message, failures);
}

// 创建指定维度的 Shape。
Shape MakeShape(const std::vector<I64>& dims) {
  Shape shape;
  shape.max_dim = static_cast<I64>(dims.size());
  for (Usize i = 0; i < dims.size(); ++i) {
    shape.dims[i] = dims[i];
    shape.pitch[i] = dims[i];
  }
  return shape;
}

// 计算维度元素数。
U64 Elements(const std::vector<I64>& dims) {
  U64 total = 1;
  for (const I64 dim : dims) {
    total *= static_cast<U64>(dim);
  }
  return total;
}

// 创建并绑定一个 Mock 设备缓冲与张量视图。
Bool CreateTensor(Backend* backend, const std::vector<I64>& dims,
                  DataType dtype, std::unique_ptr<Buffer>* buffer,
                  TensorView* view) {
  BufferDesc desc;
  desc.bytes = Elements(dims) * mobilenet::DataTypeSize(dtype);
  desc.alignment = 64;
  desc.usage = BufferUsage::kPersistent;
  if (!backend->CreateBuffer(desc, buffer).ok() || *buffer == nullptr) {
    return false;
  }
  view->data = buffer->get();
  view->offset = 0;
  view->bytes = desc.bytes;
  view->shape = MakeShape(dims);
  view->dtype = dtype;
  view->layout = mobilenet::runtime_graph::Layout::kRowMajor;
  view->device = mobilenet::runtime_graph::DeviceType::kDevice;
  return true;
}

// 把 F32 序列写成 half 张量。
void WriteHalves(const TensorView& view,
                 const std::vector<F32>& values) {
  auto* buffer = static_cast<Buffer*>(view.data);
  auto* data = static_cast<U16*>(buffer->DevicePtr());
  for (Usize i = 0; i < values.size(); ++i) {
    data[i] = FloatToHalf(values[i]);
  }
}

// 把 int64 序列写成张量。
void WriteInt64(const TensorView& view,
                const std::vector<I64>& values) {
  auto* buffer = static_cast<Buffer*>(view.data);
  auto* data = static_cast<I64*>(buffer->DevicePtr());
  for (Usize i = 0; i < values.size(); ++i) {
    data[i] = values[i];
  }
}

// 读取 half 张量为 F32 序列。
std::vector<F32> ReadHalves(const TensorView& view, Usize count) {
  const auto* buffer = static_cast<const Buffer*>(view.data);
  const auto* data =
      static_cast<const U16*>(buffer->DevicePtr());
  std::vector<F32> values(count, 0.0f);
  for (Usize i = 0; i < count; ++i) {
    values[i] = HalfToFloat(data[i]);
  }
  return values;
}

// 读取 int64 张量。
std::vector<I64> ReadInt64(const TensorView& view, Usize count) {
  const auto* buffer = static_cast<const Buffer*>(view.data);
  const auto* data = static_cast<const I64*>(buffer->DevicePtr());
  std::vector<I64> values(count, 0);
  for (Usize i = 0; i < count; ++i) {
    values[i] = data[i];
  }
  return values;
}

// 创建 Mock 后端并完成 Init。
Status CreateMockBackend(std::unique_ptr<Backend>* backend) {
  Status status = BackendRegistry::Instance().Create(
      "mock", BackendConfig{}, backend);
  if (!status.ok()) {
    return status;
  }
  return (*backend)->Init(BackendConfig{});
}

// 验证注册表与能力声明。
void TestRegistry(I32* failures) {
  std::unique_ptr<Backend> backend;
  const Status created = BackendRegistry::Instance().Create(
      "mock", BackendConfig{}, &backend);
  Expect(created.ok() && backend != nullptr, "registry creates mock",
         failures);
  if (backend != nullptr) {
    Expect(backend->Init(BackendConfig{}).ok(), "mock Init", failures);
    mobilenet::BackendCapability capability;
    Expect(backend->Capability(&capability).ok(),
           "mock Capability", failures);
    Expect(capability.supported_ops.size() == 8,
           "mock supports 8 ops", failures);
    Expect(capability.h2d && capability.d2h,
           "mock supports H2D/D2H", failures);
    Expect(backend->Deinit().ok(), "mock Deinit", failures);
  }
  std::unique_ptr<Backend> missing;
  Expect(BackendRegistry::Instance()
             .Create("__missing__", BackendConfig{}, &missing)
             .code() == ErrorCode::kUnsupportedBackend,
         "unknown backend rejected", failures);
}

// 验证生命周期状态机。
void TestLifecycle(I32* failures) {
  std::unique_ptr<Backend> backend;
  Status status = BackendRegistry::Instance().Create(
      "mock", BackendConfig{}, &backend);
  Expect(status.ok() && backend != nullptr, "lifecycle create", failures);
  if (backend == nullptr) {
    return;
  }
  Expect(backend->Init(BackendConfig{}).ok(), "lifecycle Init", failures);
  Expect(backend->Init(BackendConfig{}).code() == ErrorCode::kStateError,
         "F64 Init rejected", failures);
  Expect(backend->Deinit().ok(), "lifecycle Deinit", failures);
  Expect(backend->Deinit().ok(), "F64 Deinit idempotent", failures);
  mobilenet::BackendCapability capability;
  Expect(backend->Capability(&capability).code() ==
             ErrorCode::kStateError,
         "Capability after Deinit rejected", failures);
  BufferDesc desc;
  desc.bytes = 16;
  std::unique_ptr<Buffer> buffer;
  Expect(backend->CreateBuffer(desc, &buffer).code() ==
             ErrorCode::kStateError,
         "CreateBuffer after Deinit rejected", failures);
  Expect(backend->Synchronize(nullptr).code() == ErrorCode::kStateError,
         "Synchronize after Deinit rejected", failures);
}

// 验证 Buffer 与 H2D/D2H 往返。
void TestBuffers(I32* failures) {
  std::unique_ptr<Backend> backend;
  if (!CreateMockBackend(&backend).ok()) {
    Expect(false, "buffer backend init", failures);
    return;
  }
  BufferDesc desc;
  desc.bytes = 257;
  desc.alignment = 64;
  std::unique_ptr<Buffer> buffer;
  Expect(backend->CreateBuffer(desc, &buffer).ok() && buffer != nullptr,
         "CreateBuffer 257 bytes", failures);
  if (buffer != nullptr) {
    std::vector<U8> source(257);
    for (Usize i = 0; i < source.size(); ++i) {
      source[i] = static_cast<U8>(i);
    }
    Expect(backend->Upload(buffer->DevicePtr(), source.data(),
                           source.size(), nullptr)
               .ok(),
           "Upload", failures);
    std::vector<U8> destination(257, 0);
    Expect(backend->Download(destination.data(), buffer->DevicePtr(),
                             destination.size(), nullptr)
               .ok(),
           "Download", failures);
    Expect(std::memcmp(source.data(), destination.data(), source.size()) ==
               0,
           "Upload/Download roundtrip", failures);
    Expect(backend->DestroyBuffer(std::move(buffer)).ok(),
           "DestroyBuffer", failures);
  }

  BufferDesc zero;
  zero.bytes = 0;
  std::unique_ptr<Buffer> zero_buffer;
  Expect(backend->CreateBuffer(zero, &zero_buffer).code() ==
             ErrorCode::kInvalidArgument,
         "zero-byte buffer rejected", failures);

  BufferDesc bad_alignment;
  bad_alignment.bytes = 16;
  bad_alignment.alignment = 3;
  std::unique_ptr<Buffer> bad_buffer;
  Expect(backend->CreateBuffer(bad_alignment, &bad_buffer).code() ==
             ErrorCode::kBufferError,
         "non-power-of-two alignment rejected", failures);

  const U8 dummy_source[] = {0};
  Expect(backend->Upload(nullptr, dummy_source, 1, nullptr).code() ==
             ErrorCode::kInvalidArgument,
         "null Upload rejected", failures);
  Expect(backend->Deinit().ok(), "buffer Deinit", failures);
}

// 验证 ConvertParam 的直通与 FP32→FP16。
void TestConvertParam(I32* failures) {
  std::unique_ptr<Backend> backend;
  if (!CreateMockBackend(&backend).ok()) {
    Expect(false, "convert backend init", failures);
    return;
  }
  const F32 source[] = {1.0f, -2.5f, 0.5f};
  ParamDesc desc;
  desc.host_ptr = source;
  desc.bytes = sizeof(source);
  desc.source_dtype = DataType::kFloat32;
  desc.target_dtype = DataType::kFloat16;
  std::unique_ptr<Buffer> buffer;
  Expect(backend->ConvertParam(desc, &buffer).ok() && buffer != nullptr,
         "ConvertParam float32->float16", failures);
  if (buffer != nullptr) {
    const auto* data =
        static_cast<const U16*>(buffer->DevicePtr());
    Expect(data[0] == FloatToHalf(1.0f) &&
               data[1] == FloatToHalf(-2.5f) &&
               data[2] == FloatToHalf(0.5f),
           "ConvertParam half values", failures);
    Expect(buffer->Bytes() == 3 * sizeof(U16),
           "ConvertParam target bytes", failures);
  }

  ParamDesc unknown;
  unknown.host_ptr = source;
  unknown.bytes = sizeof(source);
  unknown.source_dtype = DataType::kFloat32;
  unknown.target_dtype = DataType::kInt8;
  std::unique_ptr<Buffer> unknown_buffer;
  Expect(backend->ConvertParam(unknown, &unknown_buffer).code() ==
             ErrorCode::kUnsupportedDtype,
         "ConvertParam unsupported dtype rejected", failures);
  Expect(backend->Deinit().ok(), "convert Deinit", failures);
}

// 创建 Kernel 并校验 key 分发。
void TestKernelCreation(I32* failures) {
  std::unique_ptr<Backend> backend;
  if (!CreateMockBackend(&backend).ok()) {
    Expect(false, "kernel backend init", failures);
    return;
  }
  const C8* keys[] = {"gather",        "rms_norm",
                        "linear",        "linear_swi_glu",
                        "per_head_rmsnorm_rope",
                        "mega_attention", "add",
                        "sample_argmax"};
  for (const C8* key : keys) {
    KernelDesc desc;
    desc.op_type = OpType::kAdd;
    desc.precision = DataType::kFloat16;
    desc.kernel_key = key;
    std::unique_ptr<Kernel> kernel;
    Expect(backend->CreateKernel(desc, &kernel).ok() && kernel != nullptr,
           std::string("CreateKernel ") + key, failures);
    Expect(backend->DestroyKernel(std::move(kernel)).ok(),
           std::string("DestroyKernel ") + key, failures);
  }
  KernelDesc unknown;
  unknown.kernel_key = "__unknown__";
  std::unique_ptr<Kernel> unknown_kernel;
  Expect(backend->CreateKernel(unknown, &unknown_kernel).code() ==
             ErrorCode::kUnsupportedOp,
         "unknown kernel_key rejected", failures);
  Expect(backend->Deinit().ok(), "kernel Deinit", failures);
}

// 验证 half 位级转换的关键边界。
void TestHalfUtils(I32* failures) {
  Expect(FloatToHalf(1.0f) == 0x3C00u, "half 1.0 bit pattern", failures);
  Expect(FloatToHalf(-2.5f) == 0xC100u, "half -2.5 bit pattern", failures);
  Expect(FloatToHalf(0.0f) == 0x0000u, "half 0.0 bit pattern", failures);
  Expect(FloatToHalf(65504.0f) == 0x7BFFu, "half max finite", failures);
  Expect(FloatToHalf(1.0e5f) == 0x7C00u, "half overflow to inf", failures);
  Expect(FloatToHalf(-1.0e5f) == 0xFC00u,
         "half negative overflow to inf", failures);
  Expect(FloatToHalf(std::numeric_limits<F32>::infinity()) == 0x7C00u,
         "half +inf", failures);
  Expect(FloatToHalf(-std::numeric_limits<F32>::infinity()) == 0xFC00u,
         "half -inf", failures);
  const U16 nan_bits =
      FloatToHalf(std::numeric_limits<F32>::quiet_NaN());
  Expect((nan_bits & 0x7C00u) == 0x7C00u && (nan_bits & 0x03FFu) != 0,
         "half NaN", failures);

  ExpectNear(HalfToFloat(0x3C00u), 1.0f, 0.0f, "half->F32 1.0",
             failures);
  ExpectNear(HalfToFloat(0xC100u), -2.5f, 0.0f, "half->F32 -2.5",
             failures);
  ExpectNear(HalfToFloat(0x0001u), 5.9604645e-8f, 1e-14f,
             "half->F32 smallest subnormal", failures);
  Expect(BFloat16ToHalf(0x3F80u) == 0x3C00u,
         "bfloat16 1.0 -> half 1.0", failures);
}

// 运行一个简单 Kernel。
Status RunKernel(Backend* backend, const C8* key, OpType op_type,
                 const std::vector<TensorView>& inputs,
                 const std::vector<TensorView>& outputs,
                 const mobilenet::runtime_graph::AttrMap& attrs) {
  KernelDesc desc;
  desc.op_type = op_type;
  desc.precision = DataType::kFloat16;
  desc.kernel_key = key;
  std::unique_ptr<Kernel> kernel;
  Status status = backend->CreateKernel(desc, &kernel);
  if (!status.ok()) {
    return status;
  }
  KernelContext context;
  context.inputs = inputs.data();
  context.input_count = static_cast<U32>(inputs.size());
  context.outputs = const_cast<TensorView*>(outputs.data());
  context.output_count = static_cast<U32>(outputs.size());
  context.attrs = &attrs;
  context.backend = backend;
  status = kernel->Execute(context);
  backend->DestroyKernel(std::move(kernel));
  return status;
}

// 验证 8 个 CPU Kernel 的基础数值。
void TestKernels(I32* failures) {
  std::unique_ptr<Backend> backend;
  if (!CreateMockBackend(&backend).ok()) {
    Expect(false, "kernel execute backend init", failures);
    return;
  }
  const mobilenet::runtime_graph::AttrMap empty_attrs;

  // Add：1+10, 2+20, 3+30, 4+40。
  {
    std::unique_ptr<Buffer> a_buffer;
    std::unique_ptr<Buffer> b_buffer;
    std::unique_ptr<Buffer> out_buffer;
    TensorView a;
    TensorView b;
    TensorView out;
    if (CreateTensor(backend.get(), {1, 2, 2}, DataType::kFloat16,
                     &a_buffer, &a) &&
        CreateTensor(backend.get(), {1, 2, 2}, DataType::kFloat16,
                     &b_buffer, &b) &&
        CreateTensor(backend.get(), {1, 2, 2}, DataType::kFloat16,
                     &out_buffer, &out)) {
      WriteHalves(a, {1.0f, 2.0f, 3.0f, 4.0f});
      WriteHalves(b, {10.0f, 20.0f, 30.0f, 40.0f});
      const Status status = RunKernel(backend.get(), "add", OpType::kAdd,
                                      {a, b}, {out}, empty_attrs);
      Expect(status.ok(), "add kernel executes", failures);
      const std::vector<F32> values = ReadHalves(out, 4);
      ExpectNear(values[0], 11.0f, 1e-3f, "add value 0", failures);
      ExpectNear(values[1], 22.0f, 1e-3f, "add value 1", failures);
      ExpectNear(values[2], 33.0f, 1e-3f, "add value 2", failures);
      ExpectNear(values[3], 44.0f, 1e-3f, "add value 3", failures);
    } else {
      Expect(false, "add buffers", failures);
    }
  }

  // Gather：ids {0,1}，weight {{1,2},{3,4}}。
  {
    std::unique_ptr<Buffer> ids_buffer;
    std::unique_ptr<Buffer> weight_buffer;
    std::unique_ptr<Buffer> out_buffer;
    TensorView ids;
    TensorView weight;
    TensorView out;
    if (CreateTensor(backend.get(), {1, 2}, DataType::kInt64, &ids_buffer,
                     &ids) &&
        CreateTensor(backend.get(), {2, 2}, DataType::kFloat16,
                     &weight_buffer, &weight) &&
        CreateTensor(backend.get(), {1, 2, 2}, DataType::kFloat16,
                     &out_buffer, &out)) {
      WriteInt64(ids, {0, 1});
      WriteHalves(weight, {1.0f, 2.0f, 3.0f, 4.0f});
      mobilenet::runtime_graph::AttrMap attrs;
      attrs["axis"] = static_cast<I64>(0);
      const Status status = RunKernel(backend.get(), "gather",
                                      OpType::kGather, {ids, weight},
                                      {out}, attrs);
      Expect(status.ok(), "gather kernel executes", failures);
      const std::vector<F32> values = ReadHalves(out, 4);
      ExpectNear(values[0], 1.0f, 0.0f, "gather value 0", failures);
      ExpectNear(values[1], 2.0f, 0.0f, "gather value 1", failures);
      ExpectNear(values[2], 3.0f, 0.0f, "gather value 2", failures);
      ExpectNear(values[3], 4.0f, 0.0f, "gather value 3", failures);
    } else {
      Expect(false, "gather buffers", failures);
    }
  }

  // RMSNorm：x {3,4}，w {1,1}。
  {
    std::unique_ptr<Buffer> x_buffer;
    std::unique_ptr<Buffer> w_buffer;
    std::unique_ptr<Buffer> out_buffer;
    TensorView x;
    TensorView w;
    TensorView out;
    if (CreateTensor(backend.get(), {1, 1, 2}, DataType::kFloat16,
                     &x_buffer, &x) &&
        CreateTensor(backend.get(), {2}, DataType::kFloat16, &w_buffer,
                     &w) &&
        CreateTensor(backend.get(), {1, 1, 2}, DataType::kFloat16,
                     &out_buffer, &out)) {
      WriteHalves(x, {3.0f, 4.0f});
      WriteHalves(w, {1.0f, 1.0f});
      mobilenet::runtime_graph::AttrMap attrs;
      attrs["eps"] = 0.0;
      attrs["normalized_shape"] = static_cast<I64>(2);
      const Status status = RunKernel(backend.get(), "rms_norm",
                                      OpType::kRmsNorm, {x, w}, {out},
                                      attrs);
      Expect(status.ok(), "rms_norm kernel executes", failures);
      const std::vector<F32> values = ReadHalves(out, 2);
      const F32 scale = 1.0f / std::sqrt(12.5f);
      ExpectNear(values[0], 3.0f * scale, 1e-3f,
                 "rms_norm value 0", failures);
      ExpectNear(values[1], 4.0f * scale, 1e-3f,
                 "rms_norm value 1", failures);
    } else {
      Expect(false, "rms_norm buffers", failures);
    }
  }

  // Linear：x {1,2}，W 单位阵。
  {
    std::unique_ptr<Buffer> x_buffer;
    std::unique_ptr<Buffer> w_buffer;
    std::unique_ptr<Buffer> out_buffer;
    TensorView x;
    TensorView weight;
    TensorView out;
    if (CreateTensor(backend.get(), {1, 1, 2}, DataType::kFloat16,
                     &x_buffer, &x) &&
        CreateTensor(backend.get(), {2, 2}, DataType::kFloat16,
                     &w_buffer, &weight) &&
        CreateTensor(backend.get(), {1, 1, 2}, DataType::kFloat16,
                     &out_buffer, &out)) {
      WriteHalves(x, {1.0f, 2.0f});
      WriteHalves(weight, {1.0f, 0.0f, 0.0f, 1.0f});
      const Status status = RunKernel(backend.get(), "linear",
                                      OpType::kLinear, {x, weight}, {out},
                                      empty_attrs);
      Expect(status.ok(), "linear kernel executes", failures);
      const std::vector<F32> values = ReadHalves(out, 2);
      ExpectNear(values[0], 1.0f, 1e-3f, "linear value 0", failures);
      ExpectNear(values[1], 2.0f, 1e-3f, "linear value 1", failures);
    } else {
      Expect(false, "linear buffers", failures);
    }
  }

  // LinearSwiGlu：x {1,2}，gate/up 单位阵。
  {
    std::unique_ptr<Buffer> x_buffer;
    std::unique_ptr<Buffer> gate_buffer;
    std::unique_ptr<Buffer> up_buffer;
    std::unique_ptr<Buffer> out_buffer;
    TensorView x;
    TensorView gate;
    TensorView up;
    TensorView out;
    if (CreateTensor(backend.get(), {1, 1, 2}, DataType::kFloat16,
                     &x_buffer, &x) &&
        CreateTensor(backend.get(), {2, 2}, DataType::kFloat16,
                     &gate_buffer, &gate) &&
        CreateTensor(backend.get(), {2, 2}, DataType::kFloat16,
                     &up_buffer, &up) &&
        CreateTensor(backend.get(), {1, 1, 2}, DataType::kFloat16,
                     &out_buffer, &out)) {
      WriteHalves(x, {1.0f, 2.0f});
      WriteHalves(gate, {1.0f, 0.0f, 0.0f, 1.0f});
      WriteHalves(up, {1.0f, 0.0f, 0.0f, 1.0f});
      const Status status = RunKernel(backend.get(), "linear_swi_glu",
                                      OpType::kLinearSwiGlu, {x, gate, up},
                                      {out}, empty_attrs);
      Expect(status.ok(), "linear_swi_glu kernel executes", failures);
      const std::vector<F32> values = ReadHalves(out, 2);
      const F32 silu_one = 1.0f / (1.0f + std::exp(-1.0f));
      const F32 silu_two = 2.0f / (1.0f + std::exp(-2.0f));
      ExpectNear(values[0], silu_one, 1e-3f,
                 "linear_swi_glu value 0", failures);
      ExpectNear(values[1], silu_two * 2.0f, 1e-3f,
                 "linear_swi_glu value 1", failures);
    } else {
      Expect(false, "linear_swi_glu buffers", failures);
    }
  }

  // PerHeadRmsNormRope：pos=0，RoPE 恒等，只剩 per-head RMSNorm。
  {
    std::unique_ptr<Buffer> x_buffer;
    std::unique_ptr<Buffer> w_buffer;
    std::unique_ptr<Buffer> norm_buffer;
    std::unique_ptr<Buffer> pos_buffer;
    std::unique_ptr<Buffer> out_buffer;
    TensorView x;
    TensorView weight;
    TensorView norm;
    TensorView positions;
    TensorView out;
    if (CreateTensor(backend.get(), {1, 1, 2}, DataType::kFloat16,
                     &x_buffer, &x) &&
        CreateTensor(backend.get(), {2, 2}, DataType::kFloat16,
                     &w_buffer, &weight) &&
        CreateTensor(backend.get(), {2}, DataType::kFloat16, &norm_buffer,
                     &norm) &&
        CreateTensor(backend.get(), {1, 1}, DataType::kInt64, &pos_buffer,
                     &positions) &&
        CreateTensor(backend.get(), {1, 1, 2}, DataType::kFloat16,
                     &out_buffer, &out)) {
      WriteHalves(x, {1.0f, 2.0f});
      WriteHalves(weight, {1.0f, 0.0f, 0.0f, 1.0f});
      WriteHalves(norm, {1.0f, 1.0f});
      WriteInt64(positions, {0});
      mobilenet::runtime_graph::AttrMap attrs;
      attrs["num_heads"] = static_cast<I64>(1);
      attrs["head_dim"] = static_cast<I64>(2);
      attrs["norm_eps"] = 0.0;
      attrs["rope_theta"] = 10000.0;
      attrs["rope_layout"] = std::string("neox");
      const Status status = RunKernel(
          backend.get(), "per_head_rmsnorm_rope",
          OpType::kLinearPerHeadRMSNormRope,
          {x, weight, norm, positions}, {out}, attrs);
      Expect(status.ok(), "per_head_rmsnorm_rope kernel executes",
             failures);
      const std::vector<F32> values = ReadHalves(out, 2);
      const F32 scale = 1.0f / std::sqrt(2.5f);
      ExpectNear(values[0], scale, 1e-3f,
                 "per_head_rmsnorm_rope value 0", failures);
      ExpectNear(values[1], 2.0f * scale, 1e-3f,
                 "per_head_rmsnorm_rope value 1", failures);
    } else {
      Expect(false, "per_head_rmsnorm_rope buffers", failures);
    }
  }

  // MegaAttention：单 key，输出应等于 V。
  {
    std::unique_ptr<Buffer> q_buffer;
    std::unique_ptr<Buffer> k_buffer;
    std::unique_ptr<Buffer> v_buffer;
    std::unique_ptr<Buffer> out_buffer;
    TensorView q;
    TensorView k;
    TensorView v;
    TensorView out;
    if (CreateTensor(backend.get(), {1, 1, 2}, DataType::kFloat16,
                     &q_buffer, &q) &&
        CreateTensor(backend.get(), {1, 1, 2}, DataType::kFloat16,
                     &k_buffer, &k) &&
        CreateTensor(backend.get(), {1, 1, 2}, DataType::kFloat16,
                     &v_buffer, &v) &&
        CreateTensor(backend.get(), {1, 1, 2}, DataType::kFloat16,
                     &out_buffer, &out)) {
      WriteHalves(q, {1.0f, 2.0f});
      WriteHalves(k, {1.0f, 2.0f});
      WriteHalves(v, {3.0f, 4.0f});
      mobilenet::runtime_graph::AttrMap attrs;
      attrs["num_heads"] = static_cast<I64>(1);
      attrs["num_kv_heads"] = static_cast<I64>(1);
      attrs["head_dim"] = static_cast<I64>(2);
      attrs["causal"] = true;
      attrs["softmax_scale"] = 1.0;
      const Status status = RunKernel(backend.get(), "mega_attention",
                                      OpType::kMegaAttention, {q, k, v},
                                      {out}, attrs);
      Expect(status.ok(), "mega_attention kernel executes", failures);
      const std::vector<F32> values = ReadHalves(out, 2);
      ExpectNear(values[0], 3.0f, 1e-3f,
                 "mega_attention value 0", failures);
      ExpectNear(values[1], 4.0f, 1e-3f,
                 "mega_attention value 1", failures);
    } else {
      Expect(false, "mega_attention buffers", failures);
    }
  }

  // Sample：logits {0.1, 0.9, 0.5}，argmax=1。
  {
    std::unique_ptr<Buffer> logits_buffer;
    std::unique_ptr<Buffer> out_buffer;
    TensorView logits;
    TensorView out;
    if (CreateTensor(backend.get(), {1, 1, 3}, DataType::kFloat16,
                     &logits_buffer, &logits) &&
        CreateTensor(backend.get(), {1, 1}, DataType::kInt64, &out_buffer,
                     &out)) {
      WriteHalves(logits, {0.1f, 0.9f, 0.5f});
      mobilenet::runtime_graph::AttrMap attrs;
      attrs["method"] = std::string("argmax");
      attrs["axis"] = static_cast<I64>(-1);
      const Status status = RunKernel(backend.get(), "sample_argmax",
                                      OpType::kSample, {logits}, {out},
                                      attrs);
      Expect(status.ok(), "sample kernel executes", failures);
      const std::vector<I64> values = ReadInt64(out, 1);
      Expect(values[0] == 1, "sample argmax value", failures);
    } else {
      Expect(false, "sample buffers", failures);
    }
  }

  Expect(backend->Deinit().ok(), "kernel execute Deinit", failures);
}

}  // namespace

I32 main() {
  I32 failures = 0;
  TestHalfUtils(&failures);
  TestRegistry(&failures);
  TestLifecycle(&failures);
  TestBuffers(&failures);
  TestConvertParam(&failures);
  TestKernelCreation(&failures);
  TestKernels(&failures);

  if (failures == 0) {
    std::cout << "ALL TESTS PASSED\n";
    return 0;
  }
  std::cout << "FAILURES: " << failures << "\n";
  return 1;
}
