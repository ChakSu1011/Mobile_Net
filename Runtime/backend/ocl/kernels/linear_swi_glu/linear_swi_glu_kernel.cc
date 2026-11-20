// 模块：OCL LinearSwiGlu Kernel 实现
//
// 职责：
//   校验 x/gate/up/out，编译 linear_swi_glu_nt，绑定参数并下发。
//
// 边界：
//   P4.3 基线；每 work-item 一个输出；不做 4×4 双累加分块。
//
// 依赖：
//   Runtime/backend/ocl/kernels/linear_swi_glu/linear_swi_glu_kernel.h、
//   Runtime/common/data_types.h、C++17 标准库。

#include "Runtime/backend/ocl/kernels/linear_swi_glu/linear_swi_glu_kernel.h"

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "Runtime/common/data_types.h"

namespace mobilenet {
namespace ocl {
namespace {

constexpr Usize kSwiGluLocalSize = 128;

Usize RoundUp(Usize value, Usize local) {
  return ((value + local - 1) / local) * local;
}

}  // namespace

LinearSwiGluOclKernel::LinearSwiGluOclKernel(
    OclContext* context, OclProgramCache* programs, std::string kernel_key)
    : OclKernel(context, programs, std::move(kernel_key)) {}

Status LinearSwiGluOclKernel::Prepare(
    const runtime_graph::KernelContext& context) {
  if (context.input_count < 3 || context.output_count < 1 ||
      context.inputs == nullptr || context.outputs == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "LinearSwiGluOclKernel: insufficient tensors");
  }
  Status status = ValidateView(context.inputs[0], DataType::kFloat16, "x");
  if (!status.ok()) {
    return status;
  }
  status = ValidateView(context.inputs[1], DataType::kFloat16, "gate");
  if (!status.ok()) {
    return status;
  }
  status = ValidateView(context.inputs[2], DataType::kFloat16, "up");
  if (!status.ok()) {
    return status;
  }
  status = ValidateView(context.outputs[0], DataType::kFloat16, "out");
  if (!status.ok()) {
    return status;
  }
  const runtime_graph::Shape& x_shape = context.inputs[0].shape;
  const runtime_graph::Shape& gate_shape = context.inputs[1].shape;
  const runtime_graph::Shape& up_shape = context.inputs[2].shape;
  if (x_shape.max_dim < 1 || gate_shape.max_dim != 2 ||
      up_shape.max_dim != 2) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "LinearSwiGluOclKernel: unexpected tensor rank");
  }
  const I64 hidden = x_shape.dims[x_shape.max_dim - 1];
  const I64 intermediate = gate_shape.dims[0];
  const I64 total = x_shape.GetShapeSize();
  if (hidden <= 0 || intermediate <= 0 || total <= 0 ||
      total % hidden != 0 || gate_shape.dims[1] != hidden ||
      up_shape.dims[0] != intermediate ||
      up_shape.dims[1] != hidden ||
      context.outputs[0].shape.GetShapeSize() !=
          (total / hidden) * intermediate ||
      total / hidden > std::numeric_limits<I32>::max() ||
      hidden > std::numeric_limits<I32>::max() ||
      intermediate > std::numeric_limits<I32>::max()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "LinearSwiGluOclKernel: invalid shape");
  }
  return Compile("linear_swi_glu_nt", "-cl-mad-enable");
}

Status LinearSwiGluOclKernel::ExecuteImpl(
    const runtime_graph::KernelContext& context) {
  if (context.input_count < 3 || context.output_count < 1) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "LinearSwiGluOclKernel: insufficient tensors");
  }
  const runtime_graph::TensorView& x = context.inputs[0];
  const runtime_graph::TensorView& gate = context.inputs[1];
  const runtime_graph::TensorView& up = context.inputs[2];
  const runtime_graph::TensorView& out = context.outputs[0];
  const I64 hidden = x.shape.dims[x.shape.max_dim - 1];
  const I64 intermediate = gate.shape.dims[0];
  const I64 total = x.shape.GetShapeSize();
  if (hidden <= 0 || intermediate <= 0 || total <= 0 ||
      total % hidden != 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "LinearSwiGluOclKernel: invalid shape");
  }
  const I64 rows = total / hidden;
  if (rows > std::numeric_limits<I32>::max() ||
      hidden > std::numeric_limits<I32>::max() ||
      intermediate > std::numeric_limits<I32>::max()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "LinearSwiGluOclKernel: shape exceeds int32");
  }

  I32 x_offset = 0;
  I32 gate_offset = 0;
  I32 up_offset = 0;
  I32 out_offset = 0;
  Status status = ElementOffset(x, DataType::kFloat16, &x_offset);
  if (!status.ok()) {
    return status;
  }
  status = ElementOffset(gate, DataType::kFloat16, &gate_offset);
  if (!status.ok()) {
    return status;
  }
  status = ElementOffset(up, DataType::kFloat16, &up_offset);
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
  status = SetMemArg(2, gate);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(3, gate_offset);
  if (!status.ok()) {
    return status;
  }
  status = SetMemArg(4, up);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(5, up_offset);
  if (!status.ok()) {
    return status;
  }
  status = SetMemArg(6, out);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(7, out_offset);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(8, static_cast<I32>(rows));
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(9, static_cast<I32>(hidden));
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(10, static_cast<I32>(intermediate));
  if (!status.ok()) {
    return status;
  }
  const Usize work = static_cast<Usize>(rows * intermediate);
  return Enqueue1D(RoundUp(work, kSwiGluLocalSize), kSwiGluLocalSize);
}

Status CreateLinearSwiGluOclKernel(OclContext* context,
                                   OclProgramCache* programs,
                                   const std::string& kernel_key,
                                   std::unique_ptr<Kernel>* kernel) {
  if (kernel == nullptr || context == nullptr || programs == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "CreateLinearSwiGluOclKernel: kernel/context/programs null");
  }
  *kernel = std::make_unique<LinearSwiGluOclKernel>(context, programs,
                                                    kernel_key);
  return Status::Ok();
}

}  // namespace ocl
}  // namespace mobilenet
