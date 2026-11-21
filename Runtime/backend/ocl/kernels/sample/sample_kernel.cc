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

// 保存 Context/ProgramCache 与 kernel_key。
SampleOclKernel::SampleOclKernel(OclContext* context,
                                 OclProgramCache* programs,
                                 std::string kernel_key)
    : OclKernel(context, programs, std::move(kernel_key)) {}

// 释放 sample_argmax_rows 额外 kernel。
SampleOclKernel::~SampleOclKernel() {
  ReleaseExtraKernel(&rows_kernel_);
}

// 加载期：校验 logits/输出，编译全行与行选择两个 entry。
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
  Status compile_status =
      Compile("sample_argmax", "-cl-mad-enable");
  if (!compile_status.ok()) {
    return compile_status;
  }
  return CompileExtra("sample_argmax_rows", "-cl-mad-enable",
                      &rows_kernel_);
}

// 执行：按 sample_rows 选择行做 argmax，未启用时处理全部有效行。
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
  // 生产执行时只对有效 token 行做 argmax；Executor 的 output_slot 必落在
  // 有效行内（packed 布局），padding 行不参与采样。
  const I64 valid_rows = EffectiveRows(context, rows);

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

  if (context.sample_rows_enabled) {
    if (context.sample_row_count <= 0) {
      // 本步没有请求产生 token，跳过采样。
      return Status::Ok();
    }
    if (rows_kernel_ == nullptr ||
        context.sample_rows_view.data == nullptr) {
      return MakeStatus(ErrorCode::kStateError,
                        "SampleOclKernel: missing sample rows kernel/view");
    }
    status = ValidateView(context.sample_rows_view, DataType::kInt32,
                          "sample_rows");
    if (!status.ok()) {
      return status;
    }
    I32 rows_offset = 0;
    status = ElementOffset(context.sample_rows_view, DataType::kInt32,
                           &rows_offset);
    if (!status.ok()) {
      return status;
    }
    status = SetMemArg(rows_kernel_, 0, logits);
    if (!status.ok()) return status;
    status = SetIntArg(rows_kernel_, 1, logits_offset);
    if (!status.ok()) return status;
    status = SetMemArg(rows_kernel_, 2, out);
    if (!status.ok()) return status;
    status = SetIntArg(rows_kernel_, 3, out_offset);
    if (!status.ok()) return status;
    status = SetMemArg(rows_kernel_, 4, context.sample_rows_view);
    if (!status.ok()) return status;
    status = SetIntArg(rows_kernel_, 5, rows_offset);
    if (!status.ok()) return status;
    status = SetIntArg(rows_kernel_, 6, context.sample_row_count);
    if (!status.ok()) return status;
    status = SetIntArg(rows_kernel_, 7, static_cast<I32>(vocab));
    if (!status.ok()) return status;
    return Enqueue1D(
        rows_kernel_,
        static_cast<Usize>(context.sample_row_count) * kSampleLocalSize,
        kSampleLocalSize);
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
  status = SetIntArg(4, static_cast<I32>(valid_rows));
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(5, static_cast<I32>(vocab));
  if (!status.ok()) {
    return status;
  }
  return Enqueue1D(static_cast<Usize>(valid_rows) * kSampleLocalSize,
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
