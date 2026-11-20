// 模块：OCL Sample Kernel 实现
//
// 职责：
//   校验 logits/输出形状，编译 sample_argmax，绑定参数并下发。
//
// 边界：
//   P4.2 基线；一行一 WG；不处理 top-k/top-p。
//
// 依赖：
//   Runtime/backend/ocl/kernels/sample/sample_kernel.h、
//   Runtime/common/data_types.h、C++17 标准库。

#include "Runtime/backend/ocl/kernels/sample/sample_kernel.h"

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "Runtime/common/data_types.h"

namespace mobilenet {
namespace ocl {
namespace {

constexpr Usize kSampleLocalSize = 256;

}  // namespace

SampleOclKernel::SampleOclKernel(OclContext* context,
                                 OclProgramCache* programs,
                                 std::string kernel_key)
    : OclKernel(context, programs, std::move(kernel_key)) {}

Status SampleOclKernel::Prepare(
    const runtime_graph::KernelContext& context) {
  if (context.input_count < 1 || context.output_count < 1 ||
      context.inputs == nullptr || context.outputs == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "SampleOclKernel: insufficient tensors");
  }
  Status status =
      ValidateView(context.inputs[0], DataType::kFloat16, "logits");
  if (!status.ok()) {
    return status;
  }
  status = ValidateView(context.outputs[0], DataType::kInt64, "out");
  if (!status.ok()) {
    return status;
  }
  const runtime_graph::Shape& logits_shape = context.inputs[0].shape;
  const runtime_graph::Shape& out_shape = context.outputs[0].shape;
  if (logits_shape.max_dim < 2 ||
      out_shape.max_dim + 1 != logits_shape.max_dim) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "SampleOclKernel: unexpected tensor rank");
  }
  const I64 total = logits_shape.GetShapeSize();
  const I64 vocab = logits_shape.dims[logits_shape.max_dim - 1];
  if (total <= 0 || vocab <= 0 || total % vocab != 0 ||
      total / vocab > std::numeric_limits<I32>::max() ||
      vocab > std::numeric_limits<I32>::max()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "SampleOclKernel: invalid shape");
  }
  return Compile("sample_argmax", "-cl-mad-enable");
}

Status SampleOclKernel::ExecuteImpl(
    const runtime_graph::KernelContext& context) {
  if (context.input_count < 1 || context.output_count < 1) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "SampleOclKernel: insufficient tensors");
  }
  const runtime_graph::TensorView& logits = context.inputs[0];
  const runtime_graph::TensorView& out = context.outputs[0];
  const I64 total = logits.shape.GetShapeSize();
  const I64 vocab = logits.shape.dims[logits.shape.max_dim - 1];
  if (total <= 0 || vocab <= 0 || total % vocab != 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "SampleOclKernel: invalid shape");
  }
  const I64 rows = total / vocab;
  if (rows > std::numeric_limits<I32>::max() ||
      vocab > std::numeric_limits<I32>::max()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "SampleOclKernel: shape exceeds int32");
  }

  I32 logits_offset = 0;
  I32 out_offset = 0;
  Status status =
      ElementOffset(logits, DataType::kFloat16, &logits_offset);
  if (!status.ok()) {
    return status;
  }
  status = ElementOffset(out, DataType::kInt64, &out_offset);
  if (!status.ok()) {
    return status;
  }

  status = SetMemArg(0, logits);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(1, logits_offset);
  if (!status.ok()) {
    return status;
  }
  status = SetMemArg(2, out);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(3, out_offset);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(4, static_cast<I32>(rows));
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(5, static_cast<I32>(vocab));
  if (!status.ok()) {
    return status;
  }
  return Enqueue1D(static_cast<Usize>(rows) * kSampleLocalSize,
                   kSampleLocalSize);
}

Status CreateSampleOclKernel(OclContext* context,
                             OclProgramCache* programs,
                             const std::string& kernel_key,
                             std::unique_ptr<Kernel>* kernel) {
  if (kernel == nullptr || context == nullptr || programs == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "CreateSampleOclKernel: kernel/context/programs must not be null");
  }
  *kernel = std::make_unique<SampleOclKernel>(context, programs, kernel_key);
  return Status::Ok();
}

}  // namespace ocl
}  // namespace mobilenet
