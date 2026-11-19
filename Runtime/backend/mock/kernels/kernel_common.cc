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
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace mock {

const MockBuffer* GetBuffer(const runtime_graph::TensorView& view) {
  return static_cast<const MockBuffer*>(view.data);
}

MockBuffer* GetMutableBuffer(const runtime_graph::TensorView& view) {
  return static_cast<MockBuffer*>(view.data);
}

const U16* HalfData(const runtime_graph::TensorView& view) {
  const MockBuffer* buffer = GetBuffer(view);
  if (buffer == nullptr) {
    return nullptr;
  }
  const auto* base = static_cast<const U8*>(buffer->Data());
  return reinterpret_cast<const U16*>(base + view.offset);
}

U16* MutableHalfData(const runtime_graph::TensorView& view) {
  MockBuffer* buffer = GetMutableBuffer(view);
  if (buffer == nullptr) {
    return nullptr;
  }
  auto* base = static_cast<U8*>(buffer->Data());
  return reinterpret_cast<U16*>(base + view.offset);
}

const I64* Int64Data(const runtime_graph::TensorView& view) {
  const MockBuffer* buffer = GetBuffer(view);
  if (buffer == nullptr) {
    return nullptr;
  }
  const auto* base = static_cast<const U8*>(buffer->Data());
  return reinterpret_cast<const I64*>(base + view.offset);
}

I64* MutableInt64Data(const runtime_graph::TensorView& view) {
  MockBuffer* buffer = GetMutableBuffer(view);
  if (buffer == nullptr) {
    return nullptr;
  }
  auto* base = static_cast<U8*>(buffer->Data());
  return reinterpret_cast<I64*>(base + view.offset);
}

std::vector<I64> Dims(const runtime_graph::Shape& shape) {
  if (shape.max_dim < 0 || shape.max_dim > runtime_graph::kMaxShapeDim) {
    return {};
  }
  std::vector<I64> dims;
  dims.reserve(static_cast<Usize>(shape.max_dim));
  for (I64 i = 0; i < shape.max_dim; ++i) {
    dims.push_back(shape.dims[static_cast<Usize>(i)]);
  }
  return dims;
}

I64 NumElements(const runtime_graph::Shape& shape) {
  const std::vector<I64> dims = Dims(shape);
  if (dims.empty()) {
    return 1;
  }
  I64 total = 1;
  for (const I64 dim : dims) {
    total *= dim;
  }
  return total;
}

F32 ReadHalf(const runtime_graph::TensorView& view, U64 index) {
  const U16* data = HalfData(view);
  if (data == nullptr) {
    return 0.0f;
  }
  return HalfToFloat(data[index]);
}

void WriteHalf(const runtime_graph::TensorView& view, U64 index,
               F32 value) {
  U16* data = MutableHalfData(view);
  if (data == nullptr) {
    return;
  }
  data[index] = FloatToHalf(value);
}

Status RequireInputCount(const runtime_graph::KernelContext& context,
                         U32 count) {
  if (context.inputs == nullptr || context.input_count < count) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "MockKernel: insufficient inputs");
  }
  return Status::Ok();
}

Status RequireOutputCount(const runtime_graph::KernelContext& context,
                          U32 count) {
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
