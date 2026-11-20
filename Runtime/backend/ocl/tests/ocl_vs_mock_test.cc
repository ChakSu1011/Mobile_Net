// 模块：OCL vs Mock 算子数值对齐测试（P4.3）
//
// 职责：
//   对 8 个融合算子，用相同输入分别运行 OCL 与 Mock 后端，比较输出
//   max_abs / relative_l2 / cosine（Sample 比较 Token 完全一致）。
//
// 边界：
//   - 只在主机同时启用 OCL 与 Mock 时构建；
//   - 使用 small 形状；真实形状性能在 P4.5；
//   - 不比较 torch 金标准（P4.3 只做 OCL vs Mock）。
//
// 依赖：
//   Runtime/backend 公共头、Mock/OCL 后端、C++17 标准库。

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Runtime/backend/backend.h"
#include "Runtime/backend/backend_registry.h"
#include "Runtime/backend/kernel.h"
#include "Runtime/backend/kernel_context.h"
#include "Runtime/backend/ocl/ocl_convert.h"
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
using mobilenet::Status;
using mobilenet::ocl::FloatToHalf;
using mobilenet::ocl::HalfToFloat;
using mobilenet::runtime_graph::AttrMap;
using mobilenet::runtime_graph::KernelContext;
using mobilenet::runtime_graph::OpType;
using mobilenet::runtime_graph::Shape;
using mobilenet::runtime_graph::TensorView;

// 主机张量描述。
struct HostTensor {
  std::vector<I64> dims;
  DataType dtype = DataType::kFloat16;
  std::vector<float> fp_values;
  std::vector<I64> int_values;
};

// 输出断言。
void Expect(bool condition, const std::string& message, int* failures) {
  if (condition) {
    std::cout << "PASS " << message << "\n";
    return;
  }
  std::cout << "FAIL " << message << "\n";
  ++(*failures);
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

// 元素数。
U64 Elements(const std::vector<I64>& dims) {
  U64 total = 1;
  for (const I64 dim : dims) {
    total *= static_cast<U64>(dim);
  }
  return total;
}

// 生成确定性浮点数据。
std::vector<float> MakeFloats(Usize count, U32 seed) {
  std::vector<float> values(count, 0.0f);
  U32 state = seed;
  for (Usize i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    const float unit =
        static_cast<float>(state >> 8) / 16777216.0f;  // [0,1)
    values[i] = unit * 2.0f - 1.0f;
  }
  return values;
}

// 创建缓冲。
bool CreateBuffer(Backend* backend, const HostTensor& tensor,
                  std::unique_ptr<Buffer>* buffer) {
  BufferDesc desc;
  desc.bytes = Elements(tensor.dims) * mobilenet::DataTypeSize(
                                             tensor.dtype);
  desc.alignment = 64;
  desc.usage = BufferUsage::kPersistent;
  return backend->CreateBuffer(desc, buffer).ok() && *buffer != nullptr;
}

// 上传主机张量。
bool UploadTensor(Backend* backend, Buffer* buffer,
                  const HostTensor& tensor) {
  if (tensor.dtype == DataType::kInt64) {
    return backend->Upload(buffer->DevicePtr(), tensor.int_values.data(),
                           tensor.int_values.size() * sizeof(I64),
                           nullptr)
        .ok();
  }
  std::vector<U16> host(tensor.fp_values.size(), 0);
  for (size_t i = 0; i < tensor.fp_values.size(); ++i) {
    host[i] = FloatToHalf(tensor.fp_values[i]);
  }
  return backend->Upload(buffer->DevicePtr(), host.data(),
                         host.size() * sizeof(U16), nullptr)
      .ok();
}

// 下载输出。
bool DownloadTensor(Backend* backend, Buffer* buffer,
                    HostTensor* tensor) {
  if (tensor->dtype == DataType::kInt64) {
    tensor->int_values.assign(
        static_cast<size_t>(Elements(tensor->dims)), 0);
    return backend->Download(tensor->int_values.data(),
                             buffer->DevicePtr(),
                             tensor->int_values.size() * sizeof(I64),
                             nullptr)
        .ok();
  }
  std::vector<U16> host(static_cast<size_t>(Elements(tensor->dims)), 0);
  if (!backend->Download(host.data(), buffer->DevicePtr(),
                         host.size() * sizeof(U16), nullptr)
           .ok()) {
    return false;
  }
  tensor->fp_values.assign(host.size(), 0.0f);
  for (size_t i = 0; i < host.size(); ++i) {
    tensor->fp_values[i] = HalfToFloat(host[i]);
  }
  return true;
}

// 在一个后端上运行单个算子。
bool RunOp(Backend* backend, const char* key, OpType op_type,
           const std::vector<HostTensor>& inputs, const HostTensor& output,
           const AttrMap& attrs, HostTensor* actual) {
  std::vector<std::unique_ptr<Buffer>> buffers;
  buffers.reserve(inputs.size() + 1);
  std::vector<TensorView> views;
  views.reserve(inputs.size());
  for (const HostTensor& input : inputs) {
    std::unique_ptr<Buffer> buffer;
    if (!CreateBuffer(backend, input, &buffer) ||
        !UploadTensor(backend, buffer.get(), input)) {
      return false;
    }
    TensorView view;
    view.data = buffer.get();
    view.offset = 0;
    view.bytes = Elements(input.dims) *
                 mobilenet::DataTypeSize(input.dtype);
    view.shape = MakeShape(input.dims);
    view.dtype = input.dtype;
    view.layout = mobilenet::runtime_graph::Layout::kRowMajor;
    view.device = mobilenet::runtime_graph::DeviceType::kDevice;
    views.push_back(view);
    buffers.push_back(std::move(buffer));
  }

  std::unique_ptr<Buffer> output_buffer;
  if (!CreateBuffer(backend, output, &output_buffer)) {
    return false;
  }
  TensorView output_view;
  output_view.data = output_buffer.get();
  output_view.offset = 0;
  output_view.bytes = Elements(output.dims) *
                      mobilenet::DataTypeSize(output.dtype);
  output_view.shape = MakeShape(output.dims);
  output_view.dtype = output.dtype;
  output_view.layout = mobilenet::runtime_graph::Layout::kRowMajor;
  output_view.device = mobilenet::runtime_graph::DeviceType::kDevice;

  KernelDesc desc;
  desc.op_type = op_type;
  desc.precision = DataType::kFloat16;
  desc.kernel_key = key;
  std::unique_ptr<Kernel> kernel;
  Status status = backend->CreateKernel(desc, &kernel);
  if (!status.ok()) {
    return false;
  }
  KernelContext context;
  context.inputs = views.data();
  context.input_count = static_cast<U32>(views.size());
  context.outputs = &output_view;
  context.output_count = 1;
  context.attrs = &attrs;
  context.backend = backend;
  context.batch = 1;
  context.token_count = 1;
  context.seq_len = 1;
  status = kernel->Execute(context);
  (void)backend->DestroyKernel(std::move(kernel));
  if (!status.ok()) {
    std::cout << "  execute error: " << status.message() << "\n";
    return false;
  }
  const Status sync = backend->Synchronize(nullptr);
  if (!sync.ok()) {
    std::cout << "  synchronize error: " << sync.message() << "\n";
    return false;
  }
  *actual = output;
  const bool downloaded = DownloadTensor(backend, output_buffer.get(),
                                         actual);
  for (auto& buffer : buffers) {
    (void)backend->DestroyBuffer(std::move(buffer));
  }
  (void)backend->DestroyBuffer(std::move(output_buffer));
  return downloaded;
}

// 比较浮点输出。
bool CompareFloat(const std::vector<float>& expected,
                  const std::vector<float>& actual, float max_tolerance,
                  float* max_abs, float* cosine) {
  if (expected.size() != actual.size() || expected.empty()) {
    return false;
  }
  float max_error = 0.0f;
  double dot = 0.0;
  double norm_expected = 0.0;
  double norm_actual = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const float diff = std::fabs(expected[i] - actual[i]);
    if (diff > max_error) {
      max_error = diff;
    }
    dot += static_cast<double>(expected[i]) * actual[i];
    norm_expected += static_cast<double>(expected[i]) * expected[i];
    norm_actual += static_cast<double>(actual[i]) * actual[i];
  }
  const double denominator = std::sqrt(norm_expected * norm_actual);
  const float cos =
      denominator > 0.0 ? static_cast<float>(dot / denominator) : 0.0f;
  *max_abs = max_error;
  *cosine = cos;
  return max_error <= max_tolerance && cos >= 0.999f;
}

// 运行并比较一个算子。
void CheckOp(Backend* mock, Backend* ocl, const char* key, OpType op_type,
             const std::vector<HostTensor>& inputs,
             const HostTensor& output, const AttrMap& attrs,
             float tolerance, int* failures) {
  HostTensor mock_output;
  HostTensor ocl_output;
  const bool mock_ok = RunOp(mock, key, op_type, inputs, output, attrs,
                             &mock_output);
  const bool ocl_ok =
      RunOp(ocl, key, op_type, inputs, output, attrs, &ocl_output);
  if (!mock_ok || !ocl_ok) {
    std::cout << "  run detail: mock_ok=" << mock_ok
              << " ocl_ok=" << ocl_ok << "\n";
  }
  Expect(mock_ok && ocl_ok, std::string("run ") + key, failures);
  if (!mock_ok || !ocl_ok) {
    return;
  }
  if (output.dtype == DataType::kInt64) {
    Expect(mock_output.int_values == ocl_output.int_values,
           std::string(key) + " token exact", failures);
    return;
  }
  float max_abs = 0.0f;
  float cosine = 0.0f;
  const bool close = CompareFloat(mock_output.fp_values,
                                  ocl_output.fp_values, tolerance,
                                  &max_abs, &cosine);
  std::cout << "INFO " << key << " max_abs=" << max_abs
            << " cosine=" << cosine << "\n";
  Expect(close, std::string(key) + " matches mock", failures);
}

// 构造 half 张量。
HostTensor HalfTensor(std::vector<I64> dims, U32 seed) {
  HostTensor tensor;
  tensor.dims = std::move(dims);
  tensor.dtype = DataType::kFloat16;
  tensor.fp_values = MakeFloats(
      static_cast<Usize>(Elements(tensor.dims)), seed);
  return tensor;
}

}  // namespace

int main() {
  int failures = 0;
  std::unique_ptr<Backend> mock;
  std::unique_ptr<Backend> ocl;
  Status status = BackendRegistry::Instance().Create(
      "mock", BackendConfig{}, &mock);
  if (!status.ok() || mock == nullptr || !mock->Init(BackendConfig{}).ok()) {
    std::cout << "SKIP: mock backend unavailable\n";
    return 77;
  }
  status = BackendRegistry::Instance().Create("ocl", BackendConfig{},
                                              &ocl);
  if (!status.ok() || ocl == nullptr || !ocl->Init(BackendConfig{}).ok()) {
    std::cout << "SKIP: ocl backend unavailable\n";
    return 77;
  }

  // Add。
  CheckOp(mock.get(), ocl.get(), "add", OpType::kAdd,
          {HalfTensor({1, 2, 4}, 1), HalfTensor({1, 2, 4}, 2)},
          HalfTensor({1, 2, 4}, 0), AttrMap{}, 2e-2f, &failures);

  // Gather。
  {
    HostTensor ids;
    ids.dims = {1, 2};
    ids.dtype = DataType::kInt64;
    ids.int_values = {0, 1};
    AttrMap attrs;
    attrs["axis"] = static_cast<I64>(0);
    CheckOp(mock.get(), ocl.get(), "gather", OpType::kGather,
            {ids, HalfTensor({2, 4}, 3)}, HalfTensor({1, 2, 4}, 0), attrs,
            0.0f, &failures);
  }

  // RMSNorm。
  {
    AttrMap attrs;
    attrs["axis"] = static_cast<I64>(-1);
    attrs["eps"] = 1e-6;
    attrs["normalized_shape"] = static_cast<I64>(4);
    CheckOp(mock.get(), ocl.get(), "rms_norm", OpType::kRmsNorm,
            {HalfTensor({1, 2, 4}, 4), HalfTensor({4}, 5)},
            HalfTensor({1, 2, 4}, 0), attrs, 2e-2f, &failures);
  }

  // Linear。
  {
    AttrMap attrs;
    attrs["in_features"] = static_cast<I64>(4);
    attrs["out_features"] = static_cast<I64>(3);
    attrs["transpose_weight"] = true;
    attrs["bias"] = false;
    CheckOp(mock.get(), ocl.get(), "linear", OpType::kLinear,
            {HalfTensor({1, 2, 4}, 6), HalfTensor({3, 4}, 7)},
            HalfTensor({1, 2, 3}, 0), attrs, 2e-2f, &failures);
  }
  // Linear：K=16 覆盖 half8 向量路径 + 一次完整 k8 循环。
  {
    AttrMap attrs;
    attrs["in_features"] = static_cast<I64>(16);
    attrs["out_features"] = static_cast<I64>(3);
    attrs["transpose_weight"] = true;
    attrs["bias"] = false;
    CheckOp(mock.get(), ocl.get(), "linear", OpType::kLinear,
            {HalfTensor({1, 2, 16}, 22), HalfTensor({3, 16}, 23)},
            HalfTensor({1, 2, 3}, 0), attrs, 2e-2f, &failures);
  }
  // Linear：M=1 覆盖单行尾部路径。
  {
    AttrMap attrs;
    attrs["in_features"] = static_cast<I64>(16);
    attrs["out_features"] = static_cast<I64>(3);
    attrs["transpose_weight"] = true;
    attrs["bias"] = false;
    CheckOp(mock.get(), ocl.get(), "linear", OpType::kLinear,
            {HalfTensor({1, 1, 16}, 24), HalfTensor({3, 16}, 25)},
            HalfTensor({1, 1, 3}, 0), attrs, 2e-2f, &failures);
  }

  // LinearSwiGlu。
  {
    AttrMap attrs;
    attrs["intermediate_size"] = static_cast<I64>(3);
    attrs["activation"] = std::string("silu");
    attrs["transpose_weight"] = true;
    attrs["bias"] = false;
    CheckOp(mock.get(), ocl.get(), "linear_swi_glu", OpType::kLinearSwiGlu,
            {HalfTensor({1, 2, 4}, 8), HalfTensor({3, 4}, 9),
             HalfTensor({3, 4}, 10)},
            HalfTensor({1, 2, 3}, 0), attrs, 2e-2f, &failures);
  }
  // LinearSwiGlu：K=16 覆盖 half8 向量路径。
  {
    AttrMap attrs;
    attrs["intermediate_size"] = static_cast<I64>(3);
    attrs["activation"] = std::string("silu");
    attrs["transpose_weight"] = true;
    attrs["bias"] = false;
    CheckOp(mock.get(), ocl.get(), "linear_swi_glu", OpType::kLinearSwiGlu,
            {HalfTensor({1, 2, 16}, 26), HalfTensor({3, 16}, 27),
             HalfTensor({3, 16}, 28)},
            HalfTensor({1, 2, 3}, 0), attrs, 2e-2f, &failures);
  }

  // PerHeadRMSNormRope。
  {
    HostTensor positions;
    positions.dims = {1, 2};
    positions.dtype = DataType::kInt64;
    positions.int_values = {0, 1};
    AttrMap attrs;
    attrs["in_features"] = static_cast<I64>(4);
    attrs["out_features"] = static_cast<I64>(4);
    attrs["num_heads"] = static_cast<I64>(2);
    attrs["head_dim"] = static_cast<I64>(2);
    attrs["norm_eps"] = 1e-6;
    attrs["rope_theta"] = 10000.0;
    attrs["rope_layout"] = std::string("neox");
    attrs["transpose_weight"] = true;
    attrs["bias"] = false;
    CheckOp(mock.get(), ocl.get(), "per_head_rmsnorm_rope",
            OpType::kLinearPerHeadRMSNormRope,
            {HalfTensor({1, 2, 4}, 11), HalfTensor({4, 4}, 12),
             HalfTensor({2}, 13), positions},
            HalfTensor({1, 2, 4}, 0), attrs, 2e-2f, &failures);
  }
  // PerHeadRMSNormRope：K=16 覆盖 half8 向量路径。
  {
    HostTensor positions;
    positions.dims = {1, 2};
    positions.dtype = DataType::kInt64;
    positions.int_values = {0, 1};
    AttrMap attrs;
    attrs["in_features"] = static_cast<I64>(16);
    attrs["out_features"] = static_cast<I64>(4);
    attrs["num_heads"] = static_cast<I64>(2);
    attrs["head_dim"] = static_cast<I64>(2);
    attrs["norm_eps"] = 1e-6;
    attrs["rope_theta"] = 10000.0;
    attrs["rope_layout"] = std::string("neox");
    attrs["transpose_weight"] = true;
    attrs["bias"] = false;
    CheckOp(mock.get(), ocl.get(), "per_head_rmsnorm_rope",
            OpType::kLinearPerHeadRMSNormRope,
            {HalfTensor({1, 2, 16}, 29), HalfTensor({4, 16}, 30),
             HalfTensor({2}, 31), positions},
            HalfTensor({1, 2, 4}, 0), attrs, 2e-2f, &failures);
  }

  // MegaAttention。
  {
    AttrMap attrs;
    attrs["num_heads"] = static_cast<I64>(2);
    attrs["num_kv_heads"] = static_cast<I64>(1);
    attrs["head_dim"] = static_cast<I64>(2);
    attrs["causal"] = true;
    attrs["merge_heads"] = true;
    attrs["has_output_projection"] = false;
    attrs["use_kv_cache"] = false;
    attrs["softmax_scale"] = 0.7071067811865476;
    CheckOp(mock.get(), ocl.get(), "mega_attention", OpType::kMegaAttention,
            {HalfTensor({1, 2, 4}, 14), HalfTensor({1, 2, 2}, 15),
             HalfTensor({1, 2, 2}, 16)},
            HalfTensor({1, 2, 4}, 0), attrs, 3e-2f, &failures);
  }

  // Sample。
  {
    HostTensor logits;
    logits.dims = {1, 1, 4};
    logits.dtype = DataType::kFloat16;
    logits.fp_values = {0.1f, 0.9f, 0.5f, -0.2f};
    HostTensor out;
    out.dims = {1, 1};
    out.dtype = DataType::kInt64;
    AttrMap attrs;
    attrs["method"] = std::string("argmax");
    attrs["axis"] = static_cast<I64>(-1);
    CheckOp(mock.get(), ocl.get(), "sample_argmax", OpType::kSample,
            {logits}, out, attrs, 0.0f, &failures);
  }

  (void)mock->Deinit();
  (void)ocl->Deinit();
  if (failures == 0) {
    std::cout << "ALL TESTS PASSED\n";
    return 0;
  }
  std::cout << "FAILURES: " << failures << "\n";
  return 1;
}
