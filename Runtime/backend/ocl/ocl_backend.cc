// 模块：OCL Backend 实现
//
// 职责：
//   实现设备/上下文/队列、Buffer、传输、权重转换、Program 缓存与同步。
//
// 边界：
//   P4.3 已接入 8 个融合算子 Kernel；CreateKernel 由 OclKernelFactory 分发。
//
// 依赖：
//   Runtime/backend/ocl/ocl_backend.h、ocl_buffer.h、ocl_convert.h、
//   ocl_utils.h、Runtime/common/data_types.h、C++17 标准库。

#include "Runtime/backend/ocl/ocl_backend.h"

#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "Graph/include/op_type.h"
#include "Runtime/backend/ocl/ocl_buffer.h"
#include "Runtime/backend/ocl/ocl_convert.h"
#include "Runtime/backend/ocl/ocl_kernel_factory.h"
#include "Runtime/backend/ocl/ocl_utils.h"
#include "Runtime/common/data_types.h"

namespace mobilenet {
namespace ocl {

OclBackend::~OclBackend() { Deinit(); }

Status OclBackend::Init(const BackendConfig& config) {
  if (initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "OclBackend::Init: already initialized");
  }
  Status status = context_.Init(config);
  if (!status.ok()) {
    return status;
  }
  std::string cache_dir;
  const auto cache_it = config.options.find("cache_dir");
  if (cache_it != config.options.end()) {
    cache_dir = cache_it->second;
  }
  status = programs_.Init(&context_, cache_dir);
  if (!status.ok()) {
    context_.Deinit();
    return status;
  }

  capability_.supported_dtypes = {DataType::kFloat16};
  // P4.3 已接入全部 8 个融合算子；Capability 声明与工厂分发一致。
  capability_.supported_ops = {
      runtime_graph::OpType::kGather,
      runtime_graph::OpType::kRmsNorm,
      runtime_graph::OpType::kLinear,
      runtime_graph::OpType::kLinearSwiGlu,
      runtime_graph::OpType::kLinearPerHeadRMSNormRope,
      runtime_graph::OpType::kMegaAttention,
      runtime_graph::OpType::kAdd,
      runtime_graph::OpType::kSample,
  };
  capability_.h2d = true;
  capability_.d2h = true;
  const OclDeviceInfo& info = context_.info();
  capability_.max_buffer_bytes =
      info.max_mem_alloc_bytes != 0
          ? info.max_mem_alloc_bytes
          : std::numeric_limits<U64>::max() / 2;
  const U64 base_alignment =
      info.mem_base_addr_align_bits >= 8
          ? static_cast<U64>(info.mem_base_addr_align_bits / 8)
          : 16;
  capability_.min_alignment = base_alignment != 0 ? base_alignment : 16;
  capability_.attention_block_size = 16;
  capability_.kv_layout = KvLayout::kNhd;

  initialized_ = true;
  return Status::Ok();
}

Status OclBackend::Deinit() {
  (void)programs_.Deinit();
  (void)context_.Deinit();
  capability_ = BackendCapability{};
  initialized_ = false;
  return Status::Ok();
}

Status OclBackend::Capability(BackendCapability* capability) const {
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "OclBackend::Capability: not initialized");
  }
  if (capability == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "OclBackend::Capability: capability must not be null");
  }
  *capability = capability_;
  return Status::Ok();
}

Status OclBackend::CreateBuffer(const BufferDesc& desc,
                                std::unique_ptr<Buffer>* buffer) {
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "OclBackend::CreateBuffer: not initialized");
  }
  if (buffer == nullptr || desc.bytes == 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "OclBackend::CreateBuffer: invalid argument");
  }
  if (desc.bytes > capability_.max_buffer_bytes) {
    return MakeStatus(
        ErrorCode::kBufferError,
        "OclBackend::CreateBuffer: buffer exceeds max_buffer_bytes");
  }
  if (desc.alignment != 0 &&
      (desc.alignment & (desc.alignment - 1)) != 0) {
    return MakeStatus(ErrorCode::kBufferError,
                      "OclBackend::CreateBuffer: alignment must be 2^n");
  }

  OclFunctions* functions = context_.functions();
  cl_int error = CL_SUCCESS;
  cl_mem mem = functions->clCreateBuffer(
      context_.context(), CL_MEM_READ_WRITE,
      static_cast<size_t>(desc.bytes), nullptr, &error);
  if (error != CL_SUCCESS || mem == nullptr) {
    return CheckCl(error, "CreateBuffer", "clCreateBuffer");
  }
  *buffer = std::make_unique<OclBuffer>(mem, desc.bytes);
  return Status::Ok();
}

Status OclBackend::DestroyBuffer(std::unique_ptr<Buffer> buffer) {
  if (buffer == nullptr) {
    return Status::Ok();
  }
  auto* ocl_buffer = static_cast<OclBuffer*>(buffer.get());
  OclFunctions* functions = context_.functions();
  if (ocl_buffer->mem() != nullptr &&
      functions->clReleaseMemObject != nullptr) {
    (void)functions->clReleaseMemObject(ocl_buffer->mem());
  }
  buffer.reset();
  return Status::Ok();
}

Status OclBackend::ConvertParam(const ParamDesc& desc,
                                std::unique_ptr<Buffer>* buffer) {
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "OclBackend::ConvertParam: not initialized");
  }
  if (buffer == nullptr || desc.host_ptr == nullptr || desc.bytes == 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "OclBackend::ConvertParam: invalid argument");
  }
  const U32 source_size = DataTypeSize(desc.source_dtype);
  const U32 target_size = DataTypeSize(desc.target_dtype);
  if (source_size == 0 || target_size == 0 ||
      desc.bytes % source_size != 0) {
    return MakeStatus(
        ErrorCode::kUnsupportedDtype,
        "OclBackend::ConvertParam: unsupported dtype or misaligned bytes");
  }
  const U64 count = desc.bytes / source_size;
  const U64 target_bytes = count * target_size;

  BufferDesc buffer_desc;
  buffer_desc.bytes = target_bytes;
  buffer_desc.alignment = 64;
  buffer_desc.usage = BufferUsage::kPersistent;
  std::unique_ptr<Buffer> target;
  Status status = CreateBuffer(buffer_desc, &target);
  if (!status.ok()) {
    return status;
  }

  std::vector<U8> converted;
  const void* host_data = desc.host_ptr;
  if (desc.source_dtype != desc.target_dtype) {
    converted.resize(static_cast<size_t>(target_bytes));
    if (desc.source_dtype == DataType::kFloat32 &&
        desc.target_dtype == DataType::kFloat16) {
      const auto* source = static_cast<const FP32*>(desc.host_ptr);
      auto* destination = reinterpret_cast<U16*>(converted.data());
      for (U64 i = 0; i < count; ++i) {
        destination[i] = FloatToHalf(source[i]);
      }
    } else if (desc.source_dtype == DataType::kBFloat16 &&
               desc.target_dtype == DataType::kFloat16) {
      const auto* source = static_cast<const U16*>(desc.host_ptr);
      auto* destination = reinterpret_cast<U16*>(converted.data());
      for (U64 i = 0; i < count; ++i) {
        destination[i] = BFloat16ToHalf(source[i]);
      }
    } else if (desc.source_dtype == DataType::kFloat16 &&
               desc.target_dtype == DataType::kFloat32) {
      const auto* source = static_cast<const U16*>(desc.host_ptr);
      auto* destination = reinterpret_cast<FP32*>(converted.data());
      for (U64 i = 0; i < count; ++i) {
        destination[i] = HalfToFloat(source[i]);
      }
    } else {
      (void)DestroyBuffer(std::move(target));
      return MakeStatus(
          ErrorCode::kUnsupportedDtype,
          "OclBackend::ConvertParam: unsupported conversion");
    }
    host_data = converted.data();
  }

  auto* ocl_buffer = static_cast<OclBuffer*>(target.get());
  OclFunctions* functions = context_.functions();
  cl_int error = CL_SUCCESS;
  error = functions->clEnqueueWriteBuffer(
      context_.queue(), ocl_buffer->mem(), CL_TRUE, 0,
      static_cast<size_t>(target_bytes), host_data, 0, nullptr, nullptr);
  if (error != CL_SUCCESS) {
    (void)DestroyBuffer(std::move(target));
    return CheckCl(error, "ConvertParam", "upload weights");
  }
  *buffer = std::move(target);
  return Status::Ok();
}

Status OclBackend::Upload(void* device_ptr, const void* host_ptr, U64 bytes,
                          Stream*) {
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "OclBackend::Upload: not initialized");
  }
  if (device_ptr == nullptr || host_ptr == nullptr || bytes == 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "OclBackend::Upload: invalid argument");
  }
  OclFunctions* functions = context_.functions();
  cl_int error = functions->clEnqueueWriteBuffer(
      context_.queue(), reinterpret_cast<cl_mem>(device_ptr), CL_TRUE, 0,
      static_cast<size_t>(bytes), host_ptr, 0, nullptr, nullptr);
  return CheckCl(error, "Upload", "clEnqueueWriteBuffer");
}

Status OclBackend::Download(void* host_ptr, const void* device_ptr,
                            U64 bytes, Stream*) {
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "OclBackend::Download: not initialized");
  }
  if (host_ptr == nullptr || device_ptr == nullptr || bytes == 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "OclBackend::Download: invalid argument");
  }
  OclFunctions* functions = context_.functions();
  cl_int error = functions->clEnqueueReadBuffer(
      context_.queue(), reinterpret_cast<cl_mem>(
                            const_cast<void*>(device_ptr)),
      CL_TRUE, 0, static_cast<size_t>(bytes), host_ptr, 0, nullptr, nullptr);
  return CheckCl(error, "Download", "clEnqueueReadBuffer");
}

Status OclBackend::CreateKernel(const KernelDesc& desc,
                                std::unique_ptr<Kernel>* kernel) {
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "OclBackend::CreateKernel: not initialized");
  }
  if (kernel == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "OclBackend::CreateKernel: kernel output must not be null");
  }
  return CreateOclKernel(desc, &context_, &programs_, kernel);
}

Status OclBackend::DestroyKernel(std::unique_ptr<Kernel> kernel) {
  kernel.reset();
  return Status::Ok();
}

Status OclBackend::Synchronize(Stream*) {
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "OclBackend::Synchronize: not initialized");
  }
  OclFunctions* functions = context_.functions();
  const cl_int error = functions->clFinish(context_.queue());
  return CheckCl(error, "Synchronize", "clFinish");
}

Status CreateOclBackend(const BackendConfig&,
                        std::unique_ptr<Backend>* backend) {
  if (backend == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "CreateOclBackend: backend output must not be null");
  }
  *backend = std::make_unique<OclBackend>();
  return Status::Ok();
}

}  // namespace ocl
}  // namespace mobilenet
