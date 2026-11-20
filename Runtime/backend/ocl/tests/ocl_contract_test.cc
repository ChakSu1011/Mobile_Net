// 模块：OCL 后端契约测试（P4.1）
//
// 职责：
//   验证 OclContext 设备查询、Backend 生命周期、Buffer 往返、ConvertParam、
//   Program 缓存（源码编译/二进制复用）与 P4.1 的 CreateKernel 行为。
//
// 边界：
//   - 无 OpenCL 设备时返回 77（CTest Skipped），不伪造通过；
//   - 不测试 8 个算子 Kernel（P4.2 起）；
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
  Expect(capability.supported_ops.size() == 8, "ocl supports 8 ops",
         failures);
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

  mobilenet::KernelDesc kernel_desc;
  kernel_desc.kernel_key = "gather";
  std::unique_ptr<mobilenet::Kernel> kernel;
  Expect(backend->CreateKernel(kernel_desc, &kernel).code() ==
             ErrorCode::kUnsupportedOp,
         "P4.1 CreateKernel unsupported", failures);

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
