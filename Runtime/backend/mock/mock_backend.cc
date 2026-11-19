// 模块：Mock 后端实现
//
// 职责：
//   实现主机 Buffer、memcpy 传输、权重 dtype 转换、CPU Kernel 创建与
//   能力声明。
//
// 边界：
//   不做设备同步、性能优化、模型包解析或图执行。
//
// 依赖：
//   Runtime/backend/mock/mock_backend.h、mock_buffer.h、half_utils.h、
//   mock/kernels/mock_kernel_factory.h、C++17 标准库。

#include "Runtime/backend/mock/mock_backend.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <new>

#include "Graph/include/op_type.h"
#include "Runtime/backend/kernel_types.h"
#include "Runtime/backend/mock/half_utils.h"
#include "Runtime/backend/mock/kernels/mock_kernel_factory.h"
#include "Runtime/backend/mock/mock_buffer.h"
#include "Runtime/common/data_types.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace mock {
namespace {

// 判断是否为 2 的幂；0 返回 false。
Bool IsPowerOfTwo(U64 value) {
  return value != 0 && (value & (value - 1)) == 0;
}

// 把源权重字节转换到目标精度，写入 target。
//
// 参数：
//   desc：源指针、字节数与源/目标 dtype；
//   target：目标缓冲，字节数已按目标精度计算。
// 返回：
//   成功返回 kOk；未知 dtype 返回 kUnsupportedDtype。
Status ConvertParamData(const ParamDesc& desc, MockBuffer* target) {
  const U32 source_size = DataTypeSize(desc.source_dtype);
  const U32 target_size = DataTypeSize(desc.target_dtype);
  if (source_size == 0 || target_size == 0) {
    return MakeStatus(ErrorCode::kUnsupportedDtype,
                      "MockBackend::ConvertParam: unsupported dtype");
  }
  const U64 count = desc.bytes / source_size;
  if (desc.source_dtype == desc.target_dtype) {
    std::memcpy(target->Data(), desc.host_ptr, desc.bytes);
    return Status::Ok();
  }
  if (desc.source_dtype == DataType::kFloat32 &&
      desc.target_dtype == DataType::kFloat16) {
    const auto* source = static_cast<const F32*>(desc.host_ptr);
    auto* destination = static_cast<U16*>(target->Data());
    for (U64 i = 0; i < count; ++i) {
      destination[i] = FloatToHalf(source[i]);
    }
    return Status::Ok();
  }
  if (desc.source_dtype == DataType::kBFloat16 &&
      desc.target_dtype == DataType::kFloat16) {
    const auto* source = static_cast<const U16*>(desc.host_ptr);
    auto* destination = static_cast<U16*>(target->Data());
    for (U64 i = 0; i < count; ++i) {
      destination[i] = BFloat16ToHalf(source[i]);
    }
    return Status::Ok();
  }
  if (desc.source_dtype == DataType::kFloat16 &&
      desc.target_dtype == DataType::kFloat32) {
    const auto* source = static_cast<const U16*>(desc.host_ptr);
    auto* destination = static_cast<F32*>(target->Data());
    for (U64 i = 0; i < count; ++i) {
      destination[i] = HalfToFloat(source[i]);
    }
    return Status::Ok();
  }
  return MakeStatus(
      ErrorCode::kUnsupportedDtype,
      "MockBackend::ConvertParam: unsupported conversion");
}

}  // namespace

MockBackend::MockBackend() {
  // 计算 Kernel 只实现 FP16 存储 + FP32 累加；BF16/FP32 仅作为
  // ConvertParam 的源精度，不进入计算图。
  capability_.supported_dtypes = {DataType::kFloat16};
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
  capability_.max_buffer_bytes =
      std::numeric_limits<U64>::max() / 2;
  capability_.min_alignment = 1;
  capability_.attention_block_size = 16;
  capability_.kv_layout = KvLayout::kNhd;
}

MockBackend::~MockBackend() { Deinit(); }

Status MockBackend::Init(const BackendConfig& config) {
  if (initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "MockBackend::Init: already initialized");
  }
  if (config.device_id < 0 || config.queue_count == 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "MockBackend::Init: invalid device or queue config");
  }
  initialized_ = true;
  return Status::Ok();
}

Status MockBackend::Deinit() {
  initialized_ = false;
  return Status::Ok();
}

Status MockBackend::Capability(BackendCapability* capability) const {
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "MockBackend::Capability: not initialized");
  }
  if (capability == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "MockBackend::Capability: capability must not be null");
  }
  *capability = capability_;
  return Status::Ok();
}

Status MockBackend::CreateBuffer(const BufferDesc& desc,
                                 std::unique_ptr<Buffer>* buffer) {
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "MockBackend::CreateBuffer: not initialized");
  }
  if (buffer == nullptr || desc.bytes == 0) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "MockBackend::CreateBuffer: invalid argument");
  }
  if (desc.bytes > capability_.max_buffer_bytes) {
    return MakeStatus(
        ErrorCode::kBufferError,
        "MockBackend::CreateBuffer: buffer exceeds max_buffer_bytes");
  }
  const U64 requested = desc.alignment == 0 ? 64 : desc.alignment;
  if (!IsPowerOfTwo(requested)) {
    return MakeStatus(
        ErrorCode::kBufferError,
        "MockBackend::CreateBuffer: alignment must be a power of two");
  }
  // 主机分配器通常要求至少指针宽度对齐；大于请求值仍满足请求。
  const U64 alignment = std::max<U64>(requested, 64);
  try {
    *buffer = std::make_unique<MockBuffer>(desc.bytes, alignment);
  } catch (const std::bad_alloc&) {
    return MakeStatus(ErrorCode::kDeviceOom,
                      "MockBackend::CreateBuffer: out of host memory");
  }
  return Status::Ok();
}

Status MockBackend::DestroyBuffer(std::unique_ptr<Buffer> buffer) {
  buffer.reset();
  return Status::Ok();
}

Status MockBackend::ConvertParam(const ParamDesc& desc,
                                 std::unique_ptr<Buffer>* buffer) {
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "MockBackend::ConvertParam: not initialized");
  }
  if (buffer == nullptr || desc.host_ptr == nullptr || desc.bytes == 0) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "MockBackend::ConvertParam: invalid argument");
  }
  const U32 source_size = DataTypeSize(desc.source_dtype);
  const U32 target_size = DataTypeSize(desc.target_dtype);
  if (source_size == 0 || target_size == 0 ||
      desc.bytes % source_size != 0) {
    return MakeStatus(
        ErrorCode::kUnsupportedDtype,
        "MockBackend::ConvertParam: unsupported dtype or misaligned bytes");
  }
  const U64 count = desc.bytes / source_size;
  const U64 target_bytes = count * target_size;
  std::unique_ptr<MockBuffer> target;
  try {
    target = std::make_unique<MockBuffer>(target_bytes, 64);
  } catch (const std::bad_alloc&) {
    return MakeStatus(ErrorCode::kDeviceOom,
                      "MockBackend::ConvertParam: out of host memory");
  }
  const Status status = ConvertParamData(desc, target.get());
  if (!status.ok()) {
    return status;
  }
  *buffer = std::move(target);
  return Status::Ok();
}

Status MockBackend::Upload(void* device_ptr, const void* host_ptr,
                           U64 bytes, Stream*) {
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "MockBackend::Upload: not initialized");
  }
  if (device_ptr == nullptr || host_ptr == nullptr || bytes == 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "MockBackend::Upload: invalid argument");
  }
  std::memcpy(device_ptr, host_ptr, static_cast<Usize>(bytes));
  return Status::Ok();
}

Status MockBackend::Download(void* host_ptr, const void* device_ptr,
                             U64 bytes, Stream*) {
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "MockBackend::Download: not initialized");
  }
  if (host_ptr == nullptr || device_ptr == nullptr || bytes == 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "MockBackend::Download: invalid argument");
  }
  std::memcpy(host_ptr, device_ptr, static_cast<Usize>(bytes));
  return Status::Ok();
}

Status MockBackend::CreateKernel(const KernelDesc& desc,
                                 std::unique_ptr<Kernel>* kernel) {
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "MockBackend::CreateKernel: not initialized");
  }
  if (kernel == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "MockBackend::CreateKernel: kernel output must not be null");
  }
  return CreateMockKernel(desc, kernel);
}

Status MockBackend::DestroyKernel(std::unique_ptr<Kernel> kernel) {
  kernel.reset();
  return Status::Ok();
}

Status MockBackend::Synchronize(Stream*) {
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "MockBackend::Synchronize: not initialized");
  }
  return Status::Ok();
}

Status CreateMockBackend(const BackendConfig&,
                         std::unique_ptr<Backend>* backend) {
  if (backend == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "CreateMockBackend: backend output must not be null");
  }
  *backend = std::make_unique<MockBackend>();
  return Status::Ok();
}

}  // namespace mock
}  // namespace mobilenet
