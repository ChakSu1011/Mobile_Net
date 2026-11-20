// 模块：OCL RMSNorm Kernel 实现
//
// 职责：
//   校验 x/w/out，读取 eps，编译 rms_norm_row，绑定参数并下发。
//
// 边界：
//   P4.3 基线；一行一 WG；不做向量化/多行 WG 调优。
//
// 依赖：
//   Runtime/backend/ocl/kernels/rms_norm/rms_norm_kernel.h、
//   Runtime/backend/ops/attr_utils.h、Runtime/common/data_types.h。

#include "Runtime/backend/ocl/kernels/rms_norm/rms_norm_kernel.h"

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "Runtime/backend/ops/attr_utils.h"
#include "Runtime/common/data_types.h"

namespace mobilenet {
namespace ocl {
namespace {

constexpr Usize kRmsNormLocalSize = 128;

}  // namespace

RmsNormOclKernel::RmsNormOclKernel(OclContext* context,
                                   OclProgramCache* programs,
                                   std::string kernel_key)
    : OclKernel(context, programs, std::move(kernel_key)) {}

Status RmsNormOclKernel::Prepare(
    const runtime_graph::KernelContext& context) {
  if (context.input_count < 2 || context.output_count < 1 ||
      context.inputs == nullptr || context.outputs == nullptr ||
      context.attrs == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "RmsNormOclKernel: missing tensors/attrs");
  }
  Status status = ValidateView(context.inputs[0], DataType::kFloat16, "x");
  if (!status.ok()) {
    return status;
  }
  status = ValidateView(context.inputs[1], DataType::kFloat16, "w");
  if (!status.ok()) {
    return status;
  }
  status = ValidateView(context.outputs[0], DataType::kFloat16, "out");
  if (!status.ok()) {
    return status;
  }
  const runtime_graph::Shape& x_shape = context.inputs[0].shape;
  const runtime_graph::Shape& w_shape = context.inputs[1].shape;
  if (x_shape.max_dim < 1 || w_shape.max_dim != 1) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "RmsNormOclKernel: unexpected tensor rank");
  }
  const I64 hidden = x_shape.dims[x_shape.max_dim - 1];
  const I64 total = x_shape.GetShapeSize();
  if (hidden <= 0 || total <= 0 || total % hidden != 0 ||
      w_shape.dims[0] != hidden ||
      context.inputs[0].shape.GetShapeSize() !=
          context.outputs[0].shape.GetShapeSize() ||
      hidden > std::numeric_limits<I32>::max()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "RmsNormOclKernel: invalid shape");
  }
  FP64 eps = 0.0;
  status = GetFloatAttr(*context.attrs, "eps", &eps);
  if (!status.ok() || eps < 0.0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "RmsNormOclKernel: invalid eps");
  }
  return Compile("rms_norm_row", "-cl-mad-enable");
}

Status RmsNormOclKernel::ExecuteImpl(
    const runtime_graph::KernelContext& context) {
  if (context.input_count < 2 || context.output_count < 1 ||
      context.attrs == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "RmsNormOclKernel: missing tensors/attrs");
  }
  const runtime_graph::TensorView& x = context.inputs[0];
  const runtime_graph::TensorView& w = context.inputs[1];
  const runtime_graph::TensorView& out = context.outputs[0];
  const I64 hidden = x.shape.dims[x.shape.max_dim - 1];
  const I64 total = x.shape.GetShapeSize();
  if (hidden <= 0 || total <= 0 || total % hidden != 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "RmsNormOclKernel: invalid shape");
  }
  const I64 rows = total / hidden;
  if (rows > std::numeric_limits<I32>::max() ||
      hidden > std::numeric_limits<I32>::max()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "RmsNormOclKernel: shape exceeds int32");
  }
  // 生产执行时只归一化本步有效 token 的行。
  const I64 valid_rows = EffectiveRows(context, rows);
  FP64 eps = 0.0;
  Status status = GetFloatAttr(*context.attrs, "eps", &eps);
  if (!status.ok()) {
    return status;
  }

  I32 x_offset = 0;
  I32 w_offset = 0;
  I32 out_offset = 0;
  status = ElementOffset(x, DataType::kFloat16, &x_offset);
  if (!status.ok()) {
    return status;
  }
  status = ElementOffset(w, DataType::kFloat16, &w_offset);
  if (!status.ok()) {
    return status;
  }
  status = ElementOffset(out, DataType::kFloat16, &out_offset);
  if (!status.ok()) {
    return status;
  }

  status = SetMemArg(0, x);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(1, x_offset);
  if (!status.ok()) {
    return status;
  }
  status = SetMemArg(2, w);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(3, w_offset);
  if (!status.ok()) {
    return status;
  }
  status = SetMemArg(4, out);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(5, out_offset);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(6, static_cast<I32>(valid_rows));
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(7, static_cast<I32>(hidden));
  if (!status.ok()) {
    return status;
  }
  status = SetFloatArg(8, static_cast<float>(eps));
  if (!status.ok()) {
    return status;
  }
  return Enqueue1D(static_cast<Usize>(valid_rows) * kRmsNormLocalSize,
                   kRmsNormLocalSize);
}

Status CreateRmsNormOclKernel(OclContext* context,
                              OclProgramCache* programs,
                              const std::string& kernel_key,
                              std::unique_ptr<Kernel>* kernel) {
  if (kernel == nullptr || context == nullptr || programs == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "CreateRmsNormOclKernel: kernel/context/programs must not be null");
  }
  *kernel = std::make_unique<RmsNormOclKernel>(context, programs, kernel_key);
  return Status::Ok();
}

}  // namespace ocl
}  // namespace mobilenet
