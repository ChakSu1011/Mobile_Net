// 模块：Mock CPU Kernel 公共工具实现
//
// 职责：
//   实现 Buffer 访问、half 读写、形状计算与 KernelContext 参数校验。
//
// 边界：
//   不做算子数学；不分配内存；不修改 tensor 描述。
//
// 依赖：
//   Runtime/backend/mock/kernels/kernel_common.h、half_utils.h。

#include "Runtime/backend/mock/kernels/kernel_common.h"

#include "Runtime/backend/mock/half_utils.h"

namespace mobilenet {
namespace mock {

const MockBuffer* GetBuffer(const runtime_graph::TensorView& view) {
  return static_cast<const MockBuffer*>(view.data);
}

MockBuffer* GetMutableBuffer(const runtime_graph::TensorView& view) {
  return static_cast<MockBuffer*>(view.data);
}

const uint16_t* HalfData(const runtime_graph::TensorView& view) {
  const MockBuffer* buffer = GetBuffer(view);
  if (buffer == nullptr) {
    return nullptr;
  }
  const auto* base = static_cast<const uint8_t*>(buffer->Data());
  return reinterpret_cast<const uint16_t*>(base + view.offset);
}

uint16_t* MutableHalfData(const runtime_graph::TensorView& view) {
  MockBuffer* buffer = GetMutableBuffer(view);
  if (buffer == nullptr) {
    return nullptr;
  }
  auto* base = static_cast<uint8_t*>(buffer->Data());
  return reinterpret_cast<uint16_t*>(base + view.offset);
}

const int64_t* Int64Data(const runtime_graph::TensorView& view) {
  const MockBuffer* buffer = GetBuffer(view);
  if (buffer == nullptr) {
    return nullptr;
  }
  const auto* base = static_cast<const uint8_t*>(buffer->Data());
  return reinterpret_cast<const int64_t*>(base + view.offset);
}

int64_t* MutableInt64Data(const runtime_graph::TensorView& view) {
  MockBuffer* buffer = GetMutableBuffer(view);
  if (buffer == nullptr) {
    return nullptr;
  }
  auto* base = static_cast<uint8_t*>(buffer->Data());
  return reinterpret_cast<int64_t*>(base + view.offset);
}

std::vector<int64_t> Dims(const runtime_graph::Shape& shape) {
  if (shape.max_dim < 0 || shape.max_dim > runtime_graph::kMaxShapeDim) {
    return {};
  }
  std::vector<int64_t> dims;
  dims.reserve(static_cast<size_t>(shape.max_dim));
  for (int64_t i = 0; i < shape.max_dim; ++i) {
    dims.push_back(shape.dims[static_cast<size_t>(i)]);
  }
  return dims;
}

int64_t NumElements(const runtime_graph::Shape& shape) {
  const std::vector<int64_t> dims = Dims(shape);
  if (dims.empty()) {
    return 1;
  }
  int64_t total = 1;
  for (const int64_t dim : dims) {
    total *= dim;
  }
  return total;
}

float ReadHalf(const runtime_graph::TensorView& view, uint64_t index) {
  const uint16_t* data = HalfData(view);
  if (data == nullptr) {
    return 0.0f;
  }
  return HalfToFloat(data[index]);
}

void WriteHalf(const runtime_graph::TensorView& view, uint64_t index,
               float value) {
  uint16_t* data = MutableHalfData(view);
  if (data == nullptr) {
    return;
  }
  data[index] = FloatToHalf(value);
}

Status RequireInputCount(const runtime_graph::KernelContext& context,
                         uint32_t count) {
  if (context.inputs == nullptr || context.input_count < count) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "MockKernel: insufficient inputs");
  }
  return Status::Ok();
}

Status RequireOutputCount(const runtime_graph::KernelContext& context,
                          uint32_t count) {
  if (context.outputs == nullptr || context.output_count < count) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "MockKernel: insufficient outputs");
  }
  return Status::Ok();
}

Status RequireAttrs(const runtime_graph::KernelContext& context) {
  if (context.attrs == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "MockKernel: attrs must not be null");
  }
  return Status::Ok();
}

}  // namespace mock
}  // namespace mobilenet
