// 模块：OCL vs Mock 算子性能基准（P4.5）
//
// 职责：
//   对 8 个融合算子使用接近 Qwen3 的真实形状，分别测量 OCL 与 Mock
//   的 Execute+Synchronize 墙钟时间，输出加速比。
//
// 边界：
//   - 只在主机同时启用 OCL 与 Mock 时构建；
//   - 大形状（lm_head）Mock 过慢时使用 M=8 或仅测 OCL；
//   - 不验证数值（P4.3 已覆盖）；性能数据写入 test_results.md。
//
// 依赖：
//   Runtime/backend 公共头、Mock/OCL 后端、C++17 标准库。

#include <chrono>
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
using mobilenet::Kernel;
using mobilenet::KernelDesc;
using mobilenet::Status;
using mobilenet::ocl::FloatToHalf;
using mobilenet::runtime_graph::AttrMap;
using mobilenet::runtime_graph::KernelContext;
using mobilenet::runtime_graph::OpType;
using mobilenet::runtime_graph::Shape;
using mobilenet::runtime_graph::TensorView;

// 主机张量。
struct HostTensor {
  std::vector<I64> dims;
  DataType dtype = DataType::kFloat16;
  std::vector<float> fp_values;
  std::vector<I64> int_values;
};

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
    values[i] = (static_cast<float>(state >> 8) / 16777216.0f) * 0.2f -
                0.1f;
  }
  return values;
}

// 创建 half 输入。
HostTensor HalfTensor(std::vector<I64> dims, U32 seed) {
  HostTensor tensor;
  tensor.dims = std::move(dims);
  tensor.dtype = DataType::kFloat16;
  tensor.fp_values = MakeFloats(
      static_cast<Usize>(Elements(tensor.dims)), seed);
  return tensor;
}

// 创建 int64 输入。
HostTensor IntTensor(std::vector<I64> dims, std::vector<I64> values) {
  HostTensor tensor;
  tensor.dims = std::move(dims);
  tensor.dtype = DataType::kInt64;
  tensor.int_values = std::move(values);
  return tensor;
}

// 创建输出描述（不需要初始化数据）。
HostTensor OutputTensor(std::vector<I64> dims, DataType dtype) {
  HostTensor tensor;
  tensor.dims = std::move(dims);
  tensor.dtype = dtype;
  return tensor;
}

// 创建缓冲。
bool CreateBuffer(Backend* backend, const HostTensor& tensor,
                  std::unique_ptr<Buffer>* buffer) {
  BufferDesc desc;
  desc.bytes = Elements(tensor.dims) *
               mobilenet::DataTypeSize(tensor.dtype);
  desc.alignment = 64;
  desc.usage = BufferUsage::kPersistent;
  return backend->CreateBuffer(desc, buffer).ok() && *buffer != nullptr;
}

// 上传输入。
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

// 运行并计时；返回每次 Execute+Synchronize 的平均毫秒。
double BenchRun(Backend* backend, const char* key, OpType op_type,
                const std::vector<HostTensor>& inputs,
                const HostTensor& output, const AttrMap& attrs,
                int warmups, int iterations) {
  std::vector<std::unique_ptr<Buffer>> buffers;
  buffers.reserve(inputs.size() + 1);
  std::vector<TensorView> views;
  views.reserve(inputs.size());
  for (const HostTensor& input : inputs) {
    std::unique_ptr<Buffer> buffer;
    if (!CreateBuffer(backend, input, &buffer) ||
        !UploadTensor(backend, buffer.get(), input)) {
      return -1.0;
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
    return -1.0;
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
  if (!backend->CreateKernel(desc, &kernel).ok()) {
    return -1.0;
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

  for (int i = 0; i < warmups; ++i) {
    (void)kernel->Execute(context);
    (void)backend->Synchronize(nullptr);
  }
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    if (!kernel->Execute(context).ok()) {
      return -1.0;
    }
    if (!backend->Synchronize(nullptr).ok()) {
      return -1.0;
    }
  }
  const auto end = std::chrono::steady_clock::now();
  const double total_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  (void)backend->DestroyKernel(std::move(kernel));
  for (auto& buffer : buffers) {
    (void)backend->DestroyBuffer(std::move(buffer));
  }
  (void)backend->DestroyBuffer(std::move(output_buffer));
  return total_ms / static_cast<double>(iterations);
}

// 打印一组基准结果。
void BenchOp(Backend* mock, Backend* ocl, const char* key, OpType op_type,
             const std::string& label,
             const std::vector<HostTensor>& inputs,
             const HostTensor& output, const AttrMap& attrs,
             int mock_iterations) {
  const double ocl_ms =
      BenchRun(ocl, key, op_type, inputs, output, attrs, 1, 3);
  double mock_ms = -1.0;
  if (mock_iterations > 0) {
    mock_ms = BenchRun(mock, key, op_type, inputs, output, attrs, 1,
                       mock_iterations);
  }
  std::cout << "BENCH " << label << " ocl_ms=" << ocl_ms;
  if (mock_ms >= 0.0) {
    std::cout << " mock_ms=" << mock_ms
              << " speedup=" << (mock_ms / ocl_ms);
  } else {
    std::cout << " mock_ms=skipped";
  }
  std::cout << "\n";
}

}  // namespace

int main() {
  std::unique_ptr<Backend> mock;
  std::unique_ptr<Backend> ocl;
  if (!BackendRegistry::Instance()
           .Create("mock", BackendConfig{}, &mock)
           .ok() ||
      mock == nullptr || !mock->Init(BackendConfig{}).ok()) {
    std::cout << "SKIP: mock backend unavailable\n";
    return 77;
  }
  if (!BackendRegistry::Instance().Create("ocl", BackendConfig{}, &ocl).ok() ||
      ocl == nullptr || !ocl->Init(BackendConfig{}).ok()) {
    std::cout << "SKIP: ocl backend unavailable\n";
    return 77;
  }

  // Add：真实 262144 元素。
  BenchOp(mock.get(), ocl.get(), "add", OpType::kAdd, "add_262144",
          {HalfTensor({1, 256, 1024}, 1),
           HalfTensor({1, 256, 1024}, 2)},
          OutputTensor({1, 256, 1024}, DataType::kFloat16), AttrMap{}, 1);

  // Gather：256 tokens、V=151936、H=1024。
  {
    AttrMap attrs;
    attrs["axis"] = static_cast<I64>(0);
    BenchOp(mock.get(), ocl.get(), "gather", OpType::kGather,
            "gather_256x151936x1024",
            {IntTensor({1, 256}, std::vector<I64>(256, 0)),
             HalfTensor({151936, 1024}, 3)},
            OutputTensor({1, 256, 1024}, DataType::kFloat16), attrs, 1);
  }

  // RMSNorm：256 行 × 1024。
  {
    AttrMap attrs;
    attrs["axis"] = static_cast<I64>(-1);
    attrs["eps"] = 1e-6;
    attrs["normalized_shape"] = static_cast<I64>(1024);
    BenchOp(mock.get(), ocl.get(), "rms_norm", OpType::kRmsNorm,
            "rms_norm_256x1024",
            {HalfTensor({1, 256, 1024}, 4), HalfTensor({1024}, 5)},
            OutputTensor({1, 256, 1024}, DataType::kFloat16), attrs, 1);
  }

  // Linear：M=256,K=1024,N=1024 与 N=3072。
  {
    AttrMap attrs;
    attrs["in_features"] = static_cast<I64>(1024);
    attrs["out_features"] = static_cast<I64>(1024);
    attrs["transpose_weight"] = true;
    attrs["bias"] = false;
    BenchOp(mock.get(), ocl.get(), "linear", OpType::kLinear,
            "linear_256x1024x1024",
            {HalfTensor({1, 256, 1024}, 6), HalfTensor({1024, 1024}, 7)},
            OutputTensor({1, 256, 1024}, DataType::kFloat16), attrs, 1);
  }
  {
    AttrMap attrs;
    attrs["in_features"] = static_cast<I64>(1024);
    attrs["out_features"] = static_cast<I64>(3072);
    attrs["transpose_weight"] = true;
    attrs["bias"] = false;
    BenchOp(mock.get(), ocl.get(), "linear", OpType::kLinear,
            "linear_256x1024x3072",
            {HalfTensor({1, 256, 1024}, 8), HalfTensor({3072, 1024}, 9)},
            OutputTensor({1, 256, 3072}, DataType::kFloat16), attrs, 1);
  }

  // LinearSwiGlu：M=256,H=1024,I=3072。
  {
    AttrMap attrs;
    attrs["intermediate_size"] = static_cast<I64>(3072);
    attrs["activation"] = std::string("silu");
    attrs["transpose_weight"] = true;
    attrs["bias"] = false;
    BenchOp(mock.get(), ocl.get(), "linear_swi_glu", OpType::kLinearSwiGlu,
            "swiglu_256x1024x3072",
            {HalfTensor({1, 256, 1024}, 10),
             HalfTensor({3072, 1024}, 11),
             HalfTensor({3072, 1024}, 12)},
            OutputTensor({1, 256, 3072}, DataType::kFloat16), attrs, 1);
  }

  // PerHeadRMSNormRope：q M=256,K=1024,N=2048。
  {
    AttrMap attrs;
    attrs["in_features"] = static_cast<I64>(1024);
    attrs["out_features"] = static_cast<I64>(2048);
    attrs["num_heads"] = static_cast<I64>(16);
    attrs["head_dim"] = static_cast<I64>(128);
    attrs["norm_eps"] = 1e-6;
    attrs["rope_theta"] = 1000000.0;
    attrs["rope_layout"] = std::string("neox");
    attrs["transpose_weight"] = true;
    attrs["bias"] = false;
    std::vector<I64> positions(256);
    for (int i = 0; i < 256; ++i) {
      positions[i] = i;
    }
    BenchOp(mock.get(), ocl.get(), "per_head_rmsnorm_rope",
            OpType::kLinearPerHeadRMSNormRope, "rope_q_256x1024x2048",
            {HalfTensor({1, 256, 1024}, 13),
             HalfTensor({2048, 1024}, 14),
             HalfTensor({128}, 15),
             IntTensor({1, 256}, std::move(positions))},
            OutputTensor({1, 256, 2048}, DataType::kFloat16), attrs, 1);
  }

  // MegaAttention：S=256,H=16,KV=8,D=128。
  {
    AttrMap attrs;
    attrs["num_heads"] = static_cast<I64>(16);
    attrs["num_kv_heads"] = static_cast<I64>(8);
    attrs["head_dim"] = static_cast<I64>(128);
    attrs["causal"] = true;
    attrs["merge_heads"] = true;
    attrs["has_output_projection"] = false;
    attrs["use_kv_cache"] = false;
    attrs["softmax_scale"] = 0.08838834764831843;
    BenchOp(mock.get(), ocl.get(), "mega_attention", OpType::kMegaAttention,
            "attention_256h16kv8d128",
            {HalfTensor({1, 256, 2048}, 16),
             HalfTensor({1, 256, 1024}, 17),
             HalfTensor({1, 256, 1024}, 18)},
            OutputTensor({1, 256, 2048}, DataType::kFloat16), attrs, 1);
  }

  // Sample：rows=256,V=151936。
  {
    AttrMap attrs;
    attrs["method"] = std::string("argmax");
    attrs["axis"] = static_cast<I64>(-1);
    BenchOp(mock.get(), ocl.get(), "sample_argmax", OpType::kSample,
            "sample_256x151936",
            {HalfTensor({1, 256, 151936}, 19)},
            OutputTensor({1, 256}, DataType::kInt64), attrs, 1);
  }

  // lm_head：M=256,K=1024,N=151936；Mock 过慢，仅测 OCL。
  {
    AttrMap attrs;
    attrs["in_features"] = static_cast<I64>(1024);
    attrs["out_features"] = static_cast<I64>(151936);
    attrs["transpose_weight"] = true;
    attrs["bias"] = false;
    BenchOp(mock.get(), ocl.get(), "linear", OpType::kLinear,
            "lm_head_256x1024x151936",
            {HalfTensor({1, 256, 1024}, 20),
             HalfTensor({151936, 1024}, 21)},
            OutputTensor({1, 256, 151936}, DataType::kFloat16), attrs, 0);
  }

  (void)mock->Deinit();
  (void)ocl->Deinit();
  return 0;
}
