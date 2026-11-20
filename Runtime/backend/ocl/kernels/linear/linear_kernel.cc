// 模块：OCL Linear Kernel 实现
//
// 职责：
//   校验 x/W/out，编译 linear_nt_4x4，绑定参数并按输出 tile 下发。
//
// 边界：
//   P4.3 基线 4×4 寄存器块；不做权重打包/local tile。
//
// 依赖：
//   Runtime/backend/ocl/kernels/linear/linear_kernel.h、
//   Runtime/common/data_types.h、C++17 标准库。

#include "Runtime/backend/ocl/kernels/linear/linear_kernel.h"

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "Runtime/common/data_types.h"

namespace mobilenet {
namespace ocl {
namespace {

constexpr Usize kLinearLocalSize = 64;
constexpr I64 kLinearTile = 4;

// 向上取整到 local 倍数。
Usize RoundUp(Usize value, Usize local) {
  return ((value + local - 1) / local) * local;
}

}  // namespace

LinearOclKernel::LinearOclKernel(OclContext* context,
                                 OclProgramCache* programs,
                                 std::string kernel_key)
    : OclKernel(context, programs, std::move(kernel_key)) {}

Status LinearOclKernel::Prepare(
    const runtime_graph::KernelContext& context) {
  if (context.input_count < 2 || context.output_count < 1 ||
      context.inputs == nullptr || context.outputs == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "LinearOclKernel: insufficient tensors");
  }
  Status status = ValidateView(context.inputs[0], DataType::kFloat16, "x");
  if (!status.ok()) {
    return status;
  }
  status = ValidateView(context.inputs[1], DataType::kFloat16, "weight");
  if (!status.ok()) {
    return status;
  }
  status = ValidateView(context.outputs[0], DataType::kFloat16, "out");
  if (!status.ok()) {
    return status;
  }
  const runtime_graph::Shape& x_shape = context.inputs[0].shape;
  const runtime_graph::Shape& w_shape = context.inputs[1].shape;
  if (x_shape.max_dim < 1 || w_shape.max_dim != 2) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "LinearOclKernel: unexpected tensor rank");
  }
  const I64 k = x_shape.dims[x_shape.max_dim - 1];
  const I64 n = w_shape.dims[0];
  const I64 total = x_shape.GetShapeSize();
  if (k <= 0 || n <= 0 || total <= 0 || total % k != 0 ||
      w_shape.dims[1] != k ||
      context.outputs[0].shape.GetShapeSize() != (total / k) * n ||
      total / k > std::numeric_limits<I32>::max() ||
      n > std::numeric_limits<I32>::max() ||
      k > std::numeric_limits<I32>::max()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "LinearOclKernel: invalid shape");
  }
  return Compile("linear_nt_4x4", "-cl-mad-enable");
}

Status LinearOclKernel::ExecuteImpl(
    const runtime_graph::KernelContext& context) {
  if (context.input_count < 2 || context.output_count < 1) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "LinearOclKernel: insufficient tensors");
  }
  const runtime_graph::TensorView& a = context.inputs[0];
  const runtime_graph::TensorView& b = context.inputs[1];
  const runtime_graph::TensorView& c = context.outputs[0];
  const I64 k = a.shape.dims[a.shape.max_dim - 1];
  const I64 n = b.shape.dims[0];
  const I64 total = a.shape.GetShapeSize();
  if (k <= 0 || n <= 0 || total <= 0 || total % k != 0 ||
      total / k > std::numeric_limits<I32>::max() ||
      n > std::numeric_limits<I32>::max() ||
      k > std::numeric_limits<I32>::max()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "LinearOclKernel: invalid shape");
  }
  const I64 m = total / k;

  I32 a_offset = 0;
  I32 b_offset = 0;
  I32 c_offset = 0;
  Status status = ElementOffset(a, DataType::kFloat16, &a_offset);
  if (!status.ok()) {
    return status;
  }
  status = ElementOffset(b, DataType::kFloat16, &b_offset);
  if (!status.ok()) {
    return status;
  }
  status = ElementOffset(c, DataType::kFloat16, &c_offset);
  if (!status.ok()) {
    return status;
  }

  status = SetMemArg(0, a);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(1, a_offset);
  if (!status.ok()) {
    return status;
  }
  status = SetMemArg(2, b);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(3, b_offset);
  if (!status.ok()) {
    return status;
  }
  status = SetMemArg(4, c);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(5, c_offset);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(6, static_cast<I32>(m));
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(7, static_cast<I32>(n));
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(8, static_cast<I32>(k));
  if (!status.ok()) {
    return status;
  }

  const I64 tiles_n = (n + kLinearTile - 1) / kLinearTile;
  const I64 tiles_m = (m + kLinearTile - 1) / kLinearTile;
  const Usize tiles = static_cast<Usize>(tiles_m * tiles_n);
  return Enqueue1D(RoundUp(tiles, kLinearLocalSize), kLinearLocalSize);
}

Status CreateLinearOclKernel(OclContext* context,
                             OclProgramCache* programs,
                             const std::string& kernel_key,
                             std::unique_ptr<Kernel>* kernel) {
  if (kernel == nullptr || context == nullptr || programs == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "CreateLinearOclKernel: kernel/context/programs must not be null");
  }
  *kernel = std::make_unique<LinearOclKernel>(context, programs, kernel_key);
  return Status::Ok();
}

}  // namespace ocl
}  // namespace mobilenet
