// 模块：OCL 后端契约测试（P4.1/P4.2）
//
// 职责：
//   验证 OclContext 设备查询、Backend 生命周期、Buffer 往返、ConvertParam、
//   Program 缓存（源码编译/二进制复用）以及 P4.2 首批 Kernel
//   （gather/add/sample_argmax）的基础数值行为。
//
// 边界：
//   - 无 OpenCL 设备时返回 77（CTest Skipped），不伪造通过；
//   - 只测试首批 3 个 Kernel；其余算子与 torch/Mock 精度对比在 P4.3–P4.5；
//   - 不测试性能（P4.5）。
//
// 依赖：
//   Runtime/backend/ocl 基础设施、CL 头、C++17 标准库。

#include <CL/cl.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "Runtime/backend/backend.h"
#include "Runtime/backend/backend_registry.h"
#include "Runtime/backend/kernel_context.h"
#include "Runtime/backend/ocl/ocl_context.h"
#include "Runtime/backend/ocl/ocl_convert.h"
#include "Runtime/backend/ocl/ocl_program_cache.h"
#include "Runtime/backend/ocl/ocl_utils.h"
#include "Utils/cpp/types.h"

namespace {

using mobilenet::Backend;
using mobilenet::BackendCapability;
using mobilenet::BackendConfig;
using mobilenet::BackendRegistry;
using mobilenet::Buffer;
using mobilenet::BufferDesc;
using mobilenet::BufferUsage;
using mobilenet::DataType;
using mobilenet::ErrorCode;
using mobilenet::Status;
using mobilenet::ocl::FloatToHalf;
using mobilenet::ocl::HalfToFloat;
using mobilenet::ocl::OclContext;
using mobilenet::ocl::OclDeviceInfo;
using mobilenet::ocl::OclFunctions;
using mobilenet::ocl::OclProgramCache;
using mobilenet::runtime_graph::AttrMap;
using mobilenet::runtime_graph::KernelContext;
using mobilenet::runtime_graph::OpType;
using mobilenet::runtime_graph::Shape;
using mobilenet::runtime_graph::TensorView;
using mobilenet::Kernel;
using mobilenet::KernelDesc;

constexpr int kSkipReturnCode = 77;

// 输出断言。
void Expect(bool condition, const std::string& message, int* failures) {
  if (condition) {
    std::cout << "PASS " << message << "\n";
    return;
  }
  std::cout << "FAIL " << message << "\n";
  ++(*failures);
}

// 判断是否应视为“无设备 skip”。
bool IsSkipStatus(const Status& status) {
  const std::string& message = status.message();
  return message.find("no OpenCL platform") != std::string::npos ||
         message.find("cannot load OpenCL library") != std::string::npos ||
         message.find("no OpenCL GPU device") != std::string::npos;
}

// 测试用缓存目录。
std::string CacheDir() {
#if defined(__ANDROID__)
  return "/data/local/tmp/mobilenet_ocl_cache";
#else
  return "./mobilenet_ocl_cache";
#endif
}

// 构造配置。
BackendConfig MakeConfig() {
  BackendConfig config;
  config.device_id = 0;
  config.queue_count = 1;
  config.options["cache_dir"] = CacheDir();
  return config;
}

// 打印设备信息。
void PrintDeviceInfo(const OclDeviceInfo& info) {
  std::cout << "device.name: " << info.device_name << "\n";
  std::cout << "device.version: " << info.device_version << "\n";
  std::cout << "device.driver: " << info.driver_version << "\n";
  std::cout << "device.opencl_c: " << info.opencl_c_version << "\n";
  std::cout << "device.cu: " << info.compute_units << "\n";
  std::cout << "device.max_clock_mhz: " << info.max_clock_mhz << "\n";
  std::cout << "device.global_bytes: " << info.global_mem_bytes << "\n";
  std::cout << "device.max_alloc_bytes: " << info.max_mem_alloc_bytes
            << "\n";
  std::cout << "device.local_bytes: " << info.local_mem_bytes << "\n";
  std::cout << "device.max_wg: " << info.max_work_group_size << "\n";
  std::cout << "device.fp16: " << (info.fp16 ? "true" : "false") << "\n";
  std::cout << "device.subgroups: " << (info.subgroups ? "true" : "false")
            << "\n";
}

// 构造 Shape。
Shape MakeShape(const std::vector<I64>& dims) {
  Shape shape;
  shape.max_dim = static_cast<I64>(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    shape.dims[i] = dims[i];
    shape.pitch[i] = dims[i];
  }
  return shape;
}

// 计算元素数。
U64 Elements(const std::vector<I64>& dims) {
  U64 total = 1;
  for (const I64 dim : dims) {
    total *= static_cast<U64>(dim);
  }
  return total;
}

// 创建 OCL 缓冲与视图。
bool CreateTensor(Backend* backend, const std::vector<I64>& dims,
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

// 上传 half 数据。
void UploadHalves(Backend* backend, Buffer* buffer,
                  const std::vector<float>& values) {
  std::vector<U16> host(values.size(), 0);
  for (size_t i = 0; i < values.size(); ++i) {
    host[i] = FloatToHalf(values[i]);
  }
  (void)backend->Upload(buffer->DevicePtr(), host.data(),
                        host.size() * sizeof(U16), nullptr);
}

// 下载 half 数据。
std::vector<float> DownloadHalves(Backend* backend, Buffer* buffer,
                                  size_t count) {
  std::vector<U16> host(count, 0);
  (void)backend->Download(host.data(), buffer->DevicePtr(),
                          host.size() * sizeof(U16), nullptr);
  std::vector<float> values(count, 0.0f);
  for (size_t i = 0; i < count; ++i) {
    values[i] = HalfToFloat(host[i]);
  }
  return values;
}

// 上传 int64 数据。
void UploadInt64(Backend* backend, Buffer* buffer,
                 const std::vector<I64>& values) {
  (void)backend->Upload(buffer->DevicePtr(), values.data(),
                        values.size() * sizeof(I64), nullptr);
}

// 下载 int64 数据。
std::vector<I64> DownloadInt64(Backend* backend, Buffer* buffer,
                               size_t count) {
  std::vector<I64> values(count, 0);
  (void)backend->Download(values.data(), buffer->DevicePtr(),
                          values.size() * sizeof(I64), nullptr);
  return values;
}

// 执行一个 OCL Kernel。
Status RunKernel(Backend* backend, const char* key, OpType op_type,
                 const std::vector<TensorView>& inputs,
                 const std::vector<TensorView>& outputs,
                 const AttrMap& attrs) {
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
  context.batch = 1;
  context.token_count = 1;
  context.seq_len = 1;
  status = kernel->Execute(context);
  (void)backend->DestroyKernel(std::move(kernel));
  return status;
}

// 验证 packed_tokens 有效行截断。
//
// 功能：
//   用 8 行 `add` 张量模拟静态 S=8、本步只有 3 个有效 token 的生产场景，
//   验证 packed_tokens=true 时只写前 3 行、padding 行保持原值；随后关闭
//   该标志，验证回退到全量 8 行计算。
//
// 参数：
//   backend：已初始化的 OCL 后端；
//   failures：失败计数，发现错误时递增。
void TestPackedTokens(Backend* backend, int* failures) {
  std::unique_ptr<Buffer> a_buffer;
  std::unique_ptr<Buffer> b_buffer;
  std::unique_ptr<Buffer> out_buffer;
  TensorView a;
  TensorView b;
  TensorView out;
  if (!CreateTensor(backend, {1, 8, 4}, DataType::kFloat16, &a_buffer, &a) ||
      !CreateTensor(backend, {1, 8, 4}, DataType::kFloat16, &b_buffer, &b) ||
      !CreateTensor(backend, {1, 8, 4}, DataType::kFloat16, &out_buffer,
                    &out)) {
    Expect(false, "packed add buffers", failures);
    return;
  }

  std::vector<float> first(32, 0.0f);
  std::vector<float> second(32, 0.0f);
  std::vector<float> zeros(32, 0.0f);
  for (int i = 0; i < 32; ++i) {
    first[i] = static_cast<float>(i + 1);
    second[i] = static_cast<float>(i + 101);
  }
  UploadHalves(backend, a_buffer.get(), first);
  UploadHalves(backend, b_buffer.get(), second);
  UploadHalves(backend, out_buffer.get(), zeros);

  KernelDesc desc;
  desc.op_type = OpType::kAdd;
  desc.precision = DataType::kFloat16;
  desc.kernel_key = "add";
  std::unique_ptr<Kernel> kernel;
  Status status = backend->CreateKernel(desc, &kernel);
  Expect(status.ok(), "packed add kernel", failures);
  if (!status.ok()) {
    return;
  }

  TensorView inputs[2] = {a, b};
  TensorView outputs[1] = {out};
  AttrMap attrs;
  KernelContext context;
  context.inputs = inputs;
  context.input_count = 2;
  context.outputs = outputs;
  context.output_count = 1;
  context.attrs = &attrs;
  context.backend = backend;
  context.batch = 1;
  context.token_count = 3;
  context.seq_len = 8;
  context.packed_tokens = true;

  status = kernel->Execute(context);
  Expect(status.ok(), "packed add executes", failures);
  (void)backend->Synchronize(nullptr);
  std::vector<float> values =
      DownloadHalves(backend, out_buffer.get(), 32);
  bool valid_rows_ok = true;
  for (int i = 0; i < 12; ++i) {
    if (values[i] != first[i] + second[i]) {
      valid_rows_ok = false;
    }
  }
  bool padding_ok = true;
  for (int i = 12; i < 32; ++i) {
    if (values[i] != 0.0f) {
      padding_ok = false;
    }
  }
  Expect(valid_rows_ok, "packed add valid rows", failures);
  Expect(padding_ok, "packed add skips padding rows", failures);

  // 关闭 packed_tokens 后应恢复全量计算（测试/基准语义）。
  context.packed_tokens = false;
  status = kernel->Execute(context);
  Expect(status.ok(), "packed add fallback executes", failures);
  (void)backend->Synchronize(nullptr);
  values = DownloadHalves(backend, out_buffer.get(), 32);
  bool full_rows_ok = true;
  for (int i = 0; i < 32; ++i) {
    if (values[i] != first[i] + second[i]) {
      full_rows_ok = false;
    }
  }
  Expect(full_rows_ok, "packed add fallback full rows", failures);
  (void)backend->DestroyKernel(std::move(kernel));
}

// 验证 Attention 的有效长度与静态行跨度解耦。
//
// 功能：
//   构造静态 S=8、有效 S=3 的 causal GQA 输入，先按全量计算，再按 packed
//   截断计算；比较有效行的输出完全一致，padding 行保持不变。
//
// 参数：
//   backend：已初始化的 OCL 后端；
//   failures：失败计数，发现错误时递增。
void TestPackedAttention(Backend* backend, int* failures) {
  std::unique_ptr<Buffer> q_buffer;
  std::unique_ptr<Buffer> k_buffer;
  std::unique_ptr<Buffer> v_buffer;
  std::unique_ptr<Buffer> out_buffer;
  TensorView q;
  TensorView k;
  TensorView v;
  TensorView out;
  if (!CreateTensor(backend, {1, 8, 8}, DataType::kFloat16, &q_buffer, &q) ||
      !CreateTensor(backend, {1, 8, 4}, DataType::kFloat16, &k_buffer, &k) ||
      !CreateTensor(backend, {1, 8, 4}, DataType::kFloat16, &v_buffer, &v) ||
      !CreateTensor(backend, {1, 8, 8}, DataType::kFloat16, &out_buffer,
                    &out)) {
    Expect(false, "packed attention buffers", failures);
    return;
  }
  std::vector<float> q_values(64, 0.0f);
  std::vector<float> k_values(32, 0.0f);
  std::vector<float> v_values(32, 0.0f);
  std::vector<float> zeros(64, 0.0f);
  for (int i = 0; i < 64; ++i) {
    q_values[i] = static_cast<float>((i % 13) - 6) * 0.125f;
    v_values[i % 32] = static_cast<float>((i % 7) - 3) * 0.25f;
  }
  for (int i = 0; i < 32; ++i) {
    k_values[i] = static_cast<float>((i % 11) - 5) * 0.125f;
  }
  UploadHalves(backend, q_buffer.get(), q_values);
  UploadHalves(backend, k_buffer.get(), k_values);
  UploadHalves(backend, v_buffer.get(), v_values);
  UploadHalves(backend, out_buffer.get(), zeros);

  KernelDesc desc;
  desc.op_type = OpType::kMegaAttention;
  desc.precision = DataType::kFloat16;
  desc.kernel_key = "mega_attention";
  std::unique_ptr<Kernel> kernel;
  Status status = backend->CreateKernel(desc, &kernel);
  Expect(status.ok(), "packed attention kernel", failures);
  if (!status.ok()) {
    return;
  }

  AttrMap attrs;
  attrs["num_heads"] = static_cast<I64>(2);
  attrs["num_kv_heads"] = static_cast<I64>(1);
  attrs["head_dim"] = static_cast<I64>(4);
  attrs["causal"] = true;
  attrs["use_kv_cache"] = false;
  attrs["softmax_scale"] = 0.5;
  TensorView inputs[3] = {q, k, v};
  TensorView outputs[1] = {out};
  KernelContext context;
  context.inputs = inputs;
  context.input_count = 3;
  context.outputs = outputs;
  context.output_count = 1;
  context.attrs = &attrs;
  context.backend = backend;
  context.batch = 1;
  context.seq_len = 8;

  status = kernel->Execute(context);
  Expect(status.ok(), "packed attention full executes", failures);
  (void)backend->Synchronize(nullptr);
  const std::vector<float> full =
      DownloadHalves(backend, out_buffer.get(), 64);

  UploadHalves(backend, out_buffer.get(), zeros);
  context.token_count = 3;
  context.packed_tokens = true;
  status = kernel->Execute(context);
  Expect(status.ok(), "packed attention truncated executes", failures);
  (void)backend->Synchronize(nullptr);
  const std::vector<float> packed =
      DownloadHalves(backend, out_buffer.get(), 64);

  bool prefix_ok = true;
  for (int i = 0; i < 24; ++i) {
    if (packed[i] != full[i]) {
      prefix_ok = false;
    }
  }
  bool padding_ok = true;
  for (int i = 24; i < 64; ++i) {
    if (packed[i] != 0.0f) {
      padding_ok = false;
    }
  }
  Expect(prefix_ok, "packed attention valid rows", failures);
  Expect(padding_ok, "packed attention skips padding rows", failures);
  (void)backend->DestroyKernel(std::move(kernel));
}

// 首批 Kernel 基础数值测试。
void TestOclKernels(Backend* backend, int* failures) {
  // Add。
  {
    std::unique_ptr<Buffer> a_buffer;
    std::unique_ptr<Buffer> b_buffer;
    std::unique_ptr<Buffer> out_buffer;
    TensorView a;
    TensorView b;
    TensorView out;
    if (CreateTensor(backend, {1, 2, 2}, DataType::kFloat16, &a_buffer,
                     &a) &&
        CreateTensor(backend, {1, 2, 2}, DataType::kFloat16, &b_buffer,
                     &b) &&
        CreateTensor(backend, {1, 2, 2}, DataType::kFloat16, &out_buffer,
                     &out)) {
      UploadHalves(backend, a_buffer.get(), {1.0f, 2.0f, 3.0f, 4.0f});
      UploadHalves(backend, b_buffer.get(), {10.0f, 20.0f, 30.0f, 40.0f});
      const Status status =
          RunKernel(backend, "add", OpType::kAdd, {a, b}, {out}, AttrMap{});
      Expect(status.ok(), "ocl add executes", failures);
      (void)backend->Synchronize(nullptr);
      const std::vector<float> values =
          DownloadHalves(backend, out_buffer.get(), 4);
      Expect(values[0] == 11.0f && values[1] == 22.0f &&
                 values[2] == 33.0f && values[3] == 44.0f,
             "ocl add values", failures);
    } else {
      Expect(false, "ocl add buffers", failures);
    }
  }

  // Add（half8 向量路径：16 个元素）。
  {
    std::unique_ptr<Buffer> a_buffer;
    std::unique_ptr<Buffer> b_buffer;
    std::unique_ptr<Buffer> out_buffer;
    TensorView a;
    TensorView b;
    TensorView out;
    if (CreateTensor(backend, {1, 1, 16}, DataType::kFloat16, &a_buffer,
                     &a) &&
        CreateTensor(backend, {1, 1, 16}, DataType::kFloat16, &b_buffer,
                     &b) &&
        CreateTensor(backend, {1, 1, 16}, DataType::kFloat16, &out_buffer,
                     &out)) {
      std::vector<float> first(16);
      std::vector<float> second(16);
      for (size_t i = 0; i < 16; ++i) {
        first[i] = static_cast<float>(i + 1);
        second[i] = static_cast<float>(i + 101);
      }
      UploadHalves(backend, a_buffer.get(), first);
      UploadHalves(backend, b_buffer.get(), second);
      const Status status =
          RunKernel(backend, "add", OpType::kAdd, {a, b}, {out}, AttrMap{});
      Expect(status.ok(), "ocl add vector executes", failures);
      (void)backend->Synchronize(nullptr);
      const std::vector<float> values =
          DownloadHalves(backend, out_buffer.get(), 16);
      bool ok = true;
      for (size_t i = 0; i < 16; ++i) {
        if (values[i] != first[i] + second[i]) {
          ok = false;
        }
      }
      Expect(ok, "ocl add vector values", failures);
    } else {
      Expect(false, "ocl add vector buffers", failures);
    }
  }

  // Gather。
  {
    std::unique_ptr<Buffer> ids_buffer;
    std::unique_ptr<Buffer> weight_buffer;
    std::unique_ptr<Buffer> out_buffer;
    TensorView ids;
    TensorView weight;
    TensorView out;
    if (CreateTensor(backend, {1, 2}, DataType::kInt64, &ids_buffer,
                     &ids) &&
        CreateTensor(backend, {2, 2}, DataType::kFloat16, &weight_buffer,
                     &weight) &&
        CreateTensor(backend, {1, 2, 2}, DataType::kFloat16, &out_buffer,
                     &out)) {
      UploadInt64(backend, ids_buffer.get(), {0, 1});
      UploadHalves(backend, weight_buffer.get(), {1.0f, 2.0f, 3.0f, 4.0f});
      AttrMap attrs;
      attrs["axis"] = static_cast<I64>(0);
      const Status status = RunKernel(backend, "gather", OpType::kGather,
                                      {ids, weight}, {out}, attrs);
      Expect(status.ok(), "ocl gather executes", failures);
      (void)backend->Synchronize(nullptr);
      const std::vector<float> values =
          DownloadHalves(backend, out_buffer.get(), 4);
      Expect(values[0] == 1.0f && values[1] == 2.0f &&
                 values[2] == 3.0f && values[3] == 4.0f,
             "ocl gather values", failures);
    } else {
      Expect(false, "ocl gather buffers", failures);
    }
  }

  // Gather（half8 向量路径：hidden=8）。
  {
    std::unique_ptr<Buffer> ids_buffer;
    std::unique_ptr<Buffer> weight_buffer;
    std::unique_ptr<Buffer> out_buffer;
    TensorView ids;
    TensorView weight;
    TensorView out;
    if (CreateTensor(backend, {1, 2}, DataType::kInt64, &ids_buffer,
                     &ids) &&
        CreateTensor(backend, {2, 8}, DataType::kFloat16, &weight_buffer,
                     &weight) &&
        CreateTensor(backend, {1, 2, 8}, DataType::kFloat16, &out_buffer,
                     &out)) {
      UploadInt64(backend, ids_buffer.get(), {0, 1});
      std::vector<float> weights(16);
      for (size_t i = 0; i < 16; ++i) {
        weights[i] = static_cast<float>(i + 1);
      }
      UploadHalves(backend, weight_buffer.get(), weights);
      AttrMap attrs;
      attrs["axis"] = static_cast<I64>(0);
      const Status status = RunKernel(backend, "gather", OpType::kGather,
                                      {ids, weight}, {out}, attrs);
      Expect(status.ok(), "ocl gather vector executes", failures);
      (void)backend->Synchronize(nullptr);
      const std::vector<float> values =
          DownloadHalves(backend, out_buffer.get(), 16);
      bool ok = true;
      for (size_t i = 0; i < 16; ++i) {
        if (values[i] != weights[i]) {
          ok = false;
        }
      }
      Expect(ok, "ocl gather vector values", failures);
    } else {
      Expect(false, "ocl gather vector buffers", failures);
    }
  }

  // Sample。
  {
    std::unique_ptr<Buffer> logits_buffer;
    std::unique_ptr<Buffer> out_buffer;
    TensorView logits;
    TensorView out;
    if (CreateTensor(backend, {1, 1, 3}, DataType::kFloat16,
                     &logits_buffer, &logits) &&
        CreateTensor(backend, {1, 1}, DataType::kInt64, &out_buffer, &out)) {
      UploadHalves(backend, logits_buffer.get(), {0.1f, 0.9f, 0.5f});
      AttrMap attrs;
      attrs["method"] = std::string("argmax");
      attrs["axis"] = static_cast<I64>(-1);
      const Status status = RunKernel(backend, "sample_argmax",
                                      OpType::kSample, {logits}, {out},
                                      attrs);
      Expect(status.ok(), "ocl sample executes", failures);
      (void)backend->Synchronize(nullptr);
      const std::vector<I64> values =
          DownloadInt64(backend, out_buffer.get(), 1);
      Expect(values[0] == 1, "ocl sample argmax", failures);
    } else {
      Expect(false, "ocl sample buffers", failures);
    }
  }

  KernelDesc unknown;
  unknown.precision = DataType::kFloat16;
  unknown.kernel_key = "__missing__";
  std::unique_ptr<Kernel> unknown_kernel;
  Expect(backend->CreateKernel(unknown, &unknown_kernel).code() ==
             ErrorCode::kUnsupportedOp,
         "ocl unknown kernel_key rejected", failures);
}

// 测试 Backend 生命周期、Buffer 与 ConvertParam。
void TestBackend(int* failures) {
  std::unique_ptr<Backend> backend;
  Status status = BackendRegistry::Instance().Create(
      "ocl", MakeConfig(), &backend);
  Expect(status.ok() && backend != nullptr, "registry creates ocl",
         failures);
  if (backend == nullptr) {
    return;
  }
  status = backend->Init(MakeConfig());
  Expect(status.ok(), "ocl Init", failures);
  if (!status.ok()) {
    return;
  }
  Expect(backend->Init(MakeConfig()).code() == ErrorCode::kStateError,
         "double Init rejected", failures);

  BackendCapability capability;
  Expect(backend->Capability(&capability).ok(), "ocl Capability",
         failures);
  Expect(capability.supported_ops.size() == 8,
         "ocl supports 8 ops", failures);
  Expect(capability.h2d && capability.d2h, "ocl supports H2D/D2H",
         failures);
  Expect(capability.max_buffer_bytes > 0, "ocl max buffer bytes",
         failures);
  Expect(capability.min_alignment > 0, "ocl min alignment", failures);

  BufferDesc desc;
  desc.bytes = 257;
  desc.alignment = 64;
  desc.usage = BufferUsage::kPersistent;
  std::unique_ptr<Buffer> buffer;
  Expect(backend->CreateBuffer(desc, &buffer).ok() && buffer != nullptr,
         "CreateBuffer 257", failures);
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

    // UploadAt/DownloadAt：在 64 字节偏移处写 16 字节切片，验证句柄
    // 不被指针加法破坏，且切片外的数据保持不变。
    const Usize patch_offset = 64;
    std::vector<U8> patch(16, 0xAB);
    Expect(backend->UploadAt(buffer->DevicePtr(), patch_offset,
                             patch.data(), patch.size(), nullptr)
               .ok(),
           "UploadAt", failures);
    std::vector<U8> slice(patch.size(), 0);
    Expect(backend->DownloadAt(slice.data(), buffer->DevicePtr(),
                               patch_offset, slice.size(), nullptr)
               .ok(),
           "DownloadAt", failures);
    Expect(std::memcmp(patch.data(), slice.data(), patch.size()) == 0,
           "UploadAt/DownloadAt slice roundtrip", failures);
    std::vector<U8> whole(source.size(), 0);
    Expect(backend->Download(whole.data(), buffer->DevicePtr(),
                             whole.size(), nullptr)
               .ok(),
           "Download after UploadAt", failures);
    Expect(std::memcmp(whole.data(), source.data(), patch_offset) == 0 &&
               std::memcmp(whole.data() + patch_offset, patch.data(),
                           patch.size()) == 0 &&
               std::memcmp(whole.data() + patch_offset + patch.size(),
                           source.data() + patch_offset + patch.size(),
                           source.size() - patch_offset - patch.size()) ==
                   0,
           "UploadAt preserves surrounding bytes", failures);
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

  const float source_values[] = {1.0f, -2.5f, 0.5f};
  mobilenet::ParamDesc param;
  param.host_ptr = source_values;
  param.bytes = sizeof(source_values);
  param.source_dtype = DataType::kFloat32;
  param.target_dtype = DataType::kFloat16;
  std::unique_ptr<Buffer> param_buffer;
  Expect(backend->ConvertParam(param, &param_buffer).ok() &&
             param_buffer != nullptr,
         "ConvertParam fp32->fp16", failures);
  if (param_buffer != nullptr) {
    std::vector<U16> halves(3, 0);
    Expect(backend->Download(halves.data(), param_buffer->DevicePtr(),
                             halves.size() * sizeof(U16), nullptr)
               .ok(),
           "download converted weights", failures);
    Expect(halves[0] == FloatToHalf(1.0f) &&
               halves[1] == FloatToHalf(-2.5f) &&
               halves[2] == FloatToHalf(0.5f),
           "converted half values", failures);
    Expect(HalfToFloat(halves[0]) == 1.0f,
           "half roundtrip 1.0", failures);
  }

  Expect(backend->Synchronize(nullptr).ok(), "Synchronize", failures);

  TestOclKernels(backend.get(), failures);
  TestPackedTokens(backend.get(), failures);
  TestPackedAttention(backend.get(), failures);

  Expect(backend->Deinit().ok(), "ocl Deinit", failures);
  Expect(backend->Deinit().ok(), "double Deinit idempotent", failures);
  Expect(backend->Capability(&capability).code() == ErrorCode::kStateError,
         "Capability after Deinit rejected", failures);
}

// 测试 ProgramCache：编译、执行、二次加载。
void TestProgramCache(OclContext* context, int* failures) {
  OclProgramCache cache;
  Status status = cache.Init(context, CacheDir());
  Expect(status.ok(), "program cache Init", failures);
  if (!status.ok()) {
    return;
  }
  const std::string source =
      "__kernel void fill(__global float* out, float value) {\n"
      "  out[get_global_id(0)] = value;\n"
      "}\n";
  cl_program program = nullptr;
  status = cache.GetOrBuild("fill", source, "-cl-mad-enable", &program);
  Expect(status.ok() && program != nullptr, "program compile", failures);
  if (!status.ok() || program == nullptr) {
    cache.Deinit();
    return;
  }

  OclFunctions* functions = context->functions();
  cl_int error = CL_SUCCESS;
  cl_kernel kernel = functions->clCreateKernel(program, "fill", &error);
  Expect(error == CL_SUCCESS && kernel != nullptr, "create fill kernel",
         failures);
  if (kernel == nullptr) {
    cache.Deinit();
    return;
  }

  constexpr Usize kCount = 16;
  cl_mem buffer = functions->clCreateBuffer(
      context->context(), CL_MEM_READ_WRITE, kCount * sizeof(float),
      nullptr, &error);
  Expect(error == CL_SUCCESS && buffer != nullptr, "create fill buffer",
         failures);
  const float value = 3.5f;
  Expect(functions->clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer) ==
             CL_SUCCESS,
         "set fill arg 0", failures);
  Expect(functions->clSetKernelArg(kernel, 1, sizeof(float), &value) ==
             CL_SUCCESS,
         "set fill arg 1", failures);
  const size_t global = kCount;
  const size_t local = kCount;
  Expect(functions->clEnqueueNDRangeKernel(
             context->queue(), kernel, 1, nullptr, &global, &local, 0,
             nullptr, nullptr) == CL_SUCCESS,
         "enqueue fill", failures);
  Expect(functions->clFinish(context->queue()) == CL_SUCCESS,
         "finish fill", failures);
  std::vector<float> host(kCount, 0.0f);
  Expect(functions->clEnqueueReadBuffer(
             context->queue(), buffer, CL_TRUE, 0, kCount * sizeof(float),
             host.data(), 0, nullptr, nullptr) == CL_SUCCESS,
         "read fill", failures);
  Expect(host[0] == value && host[kCount - 1] == value,
         "fill values", failures);

  cl_program again = nullptr;
  Expect(cache.GetOrBuild("fill", source, "-cl-mad-enable", &again).ok() &&
             again == program,
         "program cache hit", failures);

  OclProgramCache second;
  Expect(second.Init(context, CacheDir()).ok(),
         "second cache Init", failures);
  cl_program loaded = nullptr;
  Expect(second.GetOrBuild("fill", source, "-cl-mad-enable", &loaded).ok() &&
             loaded != nullptr,
         "program binary reload", failures);
  second.Deinit();

  (void)functions->clReleaseKernel(kernel);
  (void)functions->clReleaseMemObject(buffer);
  cache.Deinit();
}

}  // namespace

int main() {
  int failures = 0;
  const BackendConfig config = MakeConfig();

  OclContext probe;
  const Status probe_status = probe.Init(config);
  if (!probe_status.ok()) {
    if (IsSkipStatus(probe_status)) {
      std::cout << "SKIP: " << probe_status.message() << "\n";
      return kSkipReturnCode;
    }
    std::cout << "FAIL: " << probe_status.message() << "\n";
    return 1;
  }
  PrintDeviceInfo(probe.info());
  TestProgramCache(&probe, &failures);
  Expect(probe.Deinit().ok(), "probe Deinit", &failures);

  TestBackend(&failures);

  if (failures == 0) {
    std::cout << "ALL TESTS PASSED\n";
    return 0;
  }
  std::cout << "FAILURES: " << failures << "\n";
  return 1;
}
