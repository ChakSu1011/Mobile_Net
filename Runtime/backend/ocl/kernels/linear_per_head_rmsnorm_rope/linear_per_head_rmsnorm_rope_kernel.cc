// 模块：OCL LinearPerHeadRMSNormRope Kernel 实现
//
// 职责：
//   校验 x/W/norm/positions/out，读取 head/rope 属性，编译并下发
//   per_head_rmsnorm_rope。
//
// 边界：
//   P4.3 基线；一 (row, head) 一 WG；D ≤ 128；不预计算 cos/sin 表。
//
// 依赖：
//   Runtime/backend/ocl/kernels/linear_per_head_rmsnorm_rope/
//   linear_per_head_rmsnorm_rope_kernel.h、Runtime/backend/ops/attr_utils.h。

#include "Runtime/backend/ocl/kernels/linear_per_head_rmsnorm_rope/linear_per_head_rmsnorm_rope_kernel.h"

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "Runtime/backend/ops/attr_utils.h"
#include "Runtime/common/data_types.h"

namespace mobilenet {
namespace ocl {
namespace {

constexpr Usize kRopeLocalSize = 128;
constexpr I64 kRopeMaxHeadDim = 128;

}  // namespace

PerHeadRmsNormRopeOclKernel::PerHeadRmsNormRopeOclKernel(
    OclContext* context, OclProgramCache* programs, std::string kernel_key)
    : OclKernel(context, programs, std::move(kernel_key)) {}

Status PerHeadRmsNormRopeOclKernel::Prepare(
    const runtime_graph::KernelContext& context) {
  if (context.input_count < 4 || context.output_count < 1 ||
      context.inputs == nullptr || context.outputs == nullptr ||
      context.attrs == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "PerHeadRmsNormRopeOclKernel: missing tensors/attrs");
  }
  Status status = ValidateView(context.inputs[0], DataType::kFloat16, "x");
  if (!status.ok()) {
    return status;
  }
  status = ValidateView(context.inputs[1], DataType::kFloat16, "weight");
  if (!status.ok()) {
    return status;
  }
  status = ValidateView(context.inputs[2], DataType::kFloat16, "norm");
  if (!status.ok()) {
    return status;
  }
  status = ValidateView(context.inputs[3], DataType::kInt64, "positions");
  if (!status.ok()) {
    return status;
  }
  status = ValidateView(context.outputs[0], DataType::kFloat16, "out");
  if (!status.ok()) {
    return status;
  }
  const runtime_graph::Shape& x_shape = context.inputs[0].shape;
  const runtime_graph::Shape& w_shape = context.inputs[1].shape;
  const runtime_graph::Shape& norm_shape = context.inputs[2].shape;
  const runtime_graph::Shape& pos_shape = context.inputs[3].shape;
  if (x_shape.max_dim < 1 || w_shape.max_dim != 2 ||
      norm_shape.max_dim != 1 || pos_shape.max_dim + 1 != x_shape.max_dim) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "PerHeadRmsNormRopeOclKernel: unexpected rank");
  }
  I64 num_heads = 0;
  I64 head_dim = 0;
  status = GetIntAttr(*context.attrs, "num_heads", &num_heads);
  if (!status.ok()) {
    return status;
  }
  status = GetIntAttr(*context.attrs, "head_dim", &head_dim);
  if (!status.ok()) {
    return status;
  }
  const I64 in_features = x_shape.dims[x_shape.max_dim - 1];
  const I64 out_features = w_shape.dims[0];
  const I64 total = x_shape.GetShapeSize();
  const I64 rows = (in_features > 0 && total % in_features == 0)
                       ? total / in_features
                       : -1;
  if (num_heads <= 0 || head_dim <= 0 ||
      head_dim > kRopeMaxHeadDim || (head_dim & 1) != 0 ||
      in_features <= 0 || out_features <= 0 ||
      w_shape.dims[1] != in_features ||
      out_features != num_heads * head_dim ||
      norm_shape.dims[0] != head_dim || rows <= 0 ||
      pos_shape.GetShapeSize() != rows ||
      context.outputs[0].shape.GetShapeSize() != rows * out_features ||
      rows > std::numeric_limits<I32>::max() ||
      in_features > std::numeric_limits<I32>::max() ||
      out_features > std::numeric_limits<I32>::max() ||
      num_heads > std::numeric_limits<I32>::max()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "PerHeadRmsNormRopeOclKernel: invalid shape");
  }
  return Compile("per_head_rmsnorm_rope", "-cl-mad-enable");
}

Status PerHeadRmsNormRopeOclKernel::ExecuteImpl(
    const runtime_graph::KernelContext& context) {
  if (context.input_count < 4 || context.output_count < 1 ||
      context.attrs == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "PerHeadRmsNormRopeOclKernel: missing tensors/attrs");
  }
  const runtime_graph::TensorView& x = context.inputs[0];
  const runtime_graph::TensorView& weight = context.inputs[1];
  const runtime_graph::TensorView& norm = context.inputs[2];
  const runtime_graph::TensorView& positions = context.inputs[3];
  const runtime_graph::TensorView& out = context.outputs[0];
  I64 num_heads = 0;
  I64 head_dim = 0;
  FP64 norm_eps = 0.0;
  FP64 rope_theta = 0.0;
  Status status = GetIntAttr(*context.attrs, "num_heads", &num_heads);
  if (!status.ok()) {
    return status;
  }
  status = GetIntAttr(*context.attrs, "head_dim", &head_dim);
  if (!status.ok()) {
    return status;
  }
  status = GetFloatAttr(*context.attrs, "norm_eps", &norm_eps);
  if (!status.ok()) {
    return status;
  }
  status = GetFloatAttr(*context.attrs, "rope_theta", &rope_theta);
  if (!status.ok()) {
    return status;
  }
  const I64 in_features = x.shape.dims[x.shape.max_dim - 1];
  const I64 total = x.shape.GetShapeSize();
  if (in_features <= 0 || total <= 0 || total % in_features != 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "PerHeadRmsNormRopeOclKernel: invalid shape");
  }
  const I64 rows = total / in_features;
  if (head_dim <= 0 || head_dim > kRopeMaxHeadDim ||
      (head_dim & 1) != 0 || num_heads <= 0 ||
      rows > std::numeric_limits<I32>::max() ||
      in_features > std::numeric_limits<I32>::max() ||
      num_heads > std::numeric_limits<I32>::max() ||
      head_dim > std::numeric_limits<I32>::max()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "PerHeadRmsNormRopeOclKernel: invalid attrs/shape");
  }

  I32 x_offset = 0;
  I32 w_offset = 0;
  I32 n_offset = 0;
  I32 p_offset = 0;
  I32 out_offset = 0;
  status = ElementOffset(x, DataType::kFloat16, &x_offset);
  if (!status.ok()) {
    return status;
  }
  status = ElementOffset(weight, DataType::kFloat16, &w_offset);
  if (!status.ok()) {
    return status;
  }
  status = ElementOffset(norm, DataType::kFloat16, &n_offset);
  if (!status.ok()) {
    return status;
  }
  status = ElementOffset(positions, DataType::kInt64, &p_offset);
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
  status = SetMemArg(2, weight);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(3, w_offset);
  if (!status.ok()) {
    return status;
  }
  status = SetMemArg(4, norm);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(5, n_offset);
  if (!status.ok()) {
    return status;
  }
  status = SetMemArg(6, positions);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(7, p_offset);
  if (!status.ok()) {
    return status;
  }
  status = SetMemArg(8, out);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(9, out_offset);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(10, static_cast<I32>(rows));
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(11, static_cast<I32>(num_heads));
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(12, static_cast<I32>(head_dim));
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(13, static_cast<I32>(in_features));
  if (!status.ok()) {
    return status;
  }
  status = SetFloatArg(14, static_cast<float>(norm_eps));
  if (!status.ok()) {
    return status;
  }
  status = SetFloatArg(15, static_cast<float>(rope_theta));
  if (!status.ok()) {
    return status;
  }
  return Enqueue1D(static_cast<Usize>(rows) * num_heads * kRopeLocalSize,
                   kRopeLocalSize);
}

Status CreatePerHeadRmsNormRopeOclKernel(
    OclContext* context, OclProgramCache* programs,
    const std::string& kernel_key, std::unique_ptr<Kernel>* kernel) {
  if (kernel == nullptr || context == nullptr || programs == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "CreatePerHeadRmsNormRopeOclKernel: null argument");
  }
  *kernel = std::make_unique<PerHeadRmsNormRopeOclKernel>(
      context, programs, kernel_key);
  return Status::Ok();
}

}  // namespace ocl
}  // namespace mobilenet
