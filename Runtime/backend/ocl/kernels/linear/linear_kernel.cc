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

// 保存 Context/ProgramCache 与 kernel_key。
LinearOclKernel::LinearOclKernel(OclContext* context,
                                 OclProgramCache* programs,
                                 std::string kernel_key)
    : OclKernel(context, programs, std::move(kernel_key)) {}

// 释放备用 4×4 entry。
LinearOclKernel::~LinearOclKernel() {
  ReleaseExtraKernel(&alt_kernel_);
}

// 加载期：校验形状，编译 4×8 主 entry 与 4×4 备用 entry。
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
  sample_rows_mode_ = false;
  if (context.attrs != nullptr &&
      context.attrs->find("sample_rows") != context.attrs->end()) {
    const auto* value = std::get_if<Bool>(
        &context.attrs->at("sample_rows"));
    if (value != nullptr) {
      sample_rows_mode_ = *value;
    }
  }
  if (sample_rows_mode_) {
    return Compile("linear_nt_rows", "-cl-mad-enable");
  }
  // FP16 累加下 4×8 的 32 个 half 累加器会导致寄存器溢出，实测比 4×4
  // 慢约 8–10 倍；因此统一使用 4×4 + half8 装载。
  return Compile("linear_nt_4x4", "-cl-mad-enable");
}

// 执行：LM Head 行选择 / prefill 4×8 / decode 4×4 三条路径。
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
  // 生产执行时只计算本步有效 token 的行，跳过静态 padding 行。
  const I64 valid_m = EffectiveRows(context, m);

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

  if (sample_rows_mode_) {
    if (!context.sample_rows_enabled) {
      return MakeStatus(
          ErrorCode::kStateError,
          "LinearOclKernel: sample_rows attr requires enabled row list");
    }
    if (context.sample_row_count <= 0) {
      // 本步没有请求产生 token（partial prefill chunk），无需计算 LM Head。
      return Status::Ok();
    }
    if (context.sample_rows_view.data == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "LinearOclKernel: missing sample rows view");
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
    status = SetMemArg(0, a);
    if (!status.ok()) return status;
    status = SetIntArg(1, a_offset);
    if (!status.ok()) return status;
    status = SetMemArg(2, b);
    if (!status.ok()) return status;
    status = SetIntArg(3, b_offset);
    if (!status.ok()) return status;
    status = SetMemArg(4, c);
    if (!status.ok()) return status;
    status = SetIntArg(5, c_offset);
    if (!status.ok()) return status;
    status = SetMemArg(6, context.sample_rows_view);
    if (!status.ok()) return status;
    status = SetIntArg(7, rows_offset);
    if (!status.ok()) return status;
    status = SetIntArg(8, context.sample_row_count);
    if (!status.ok()) return status;
    status = SetIntArg(9, static_cast<I32>(n));
    if (!status.ok()) return status;
    status = SetIntArg(10, static_cast<I32>(k));
    if (!status.ok()) return status;
    const I64 tiles_n = (n + kLinearTile - 1) / kLinearTile;
    const I64 groups =
        (static_cast<I64>(context.sample_row_count) + kLinearTile - 1) /
        kLinearTile;
    const Usize work =
        static_cast<Usize>(groups) * static_cast<Usize>(tiles_n);
    return Enqueue1D(work, 1);
  }

  // 统一使用 4×4 入口；FP16 累加下 4×8 会寄存器溢出（见 Prepare 注释）。
  cl_kernel active = kernel_;
  if (active == nullptr) {
    return MakeStatus(ErrorCode::kStateError,
                      "LinearOclKernel: no active kernel");
  }
  const I32 n_tile = 4;
  status = SetMemArg(active, 0, a);
  if (!status.ok()) return status;
  status = SetIntArg(active, 1, a_offset);
  if (!status.ok()) return status;
  status = SetMemArg(active, 2, b);
  if (!status.ok()) return status;
  status = SetIntArg(active, 3, b_offset);
  if (!status.ok()) return status;
  status = SetMemArg(active, 4, c);
  if (!status.ok()) return status;
  status = SetIntArg(active, 5, c_offset);
  if (!status.ok()) return status;
  status = SetIntArg(active, 6, static_cast<I32>(valid_m));
  if (!status.ok()) return status;
  status = SetIntArg(active, 7, static_cast<I32>(n));
  if (!status.ok()) return status;
  status = SetIntArg(active, 8, static_cast<I32>(k));
  if (!status.ok()) return status;

  const I64 tiles_n =
      (n + static_cast<I64>(n_tile) - 1) / static_cast<I64>(n_tile);
  const I64 tiles_m = (valid_m + kLinearTile - 1) / kLinearTile;
  const Usize tiles = static_cast<Usize>(tiles_m * tiles_n);
  return Enqueue1D(active, RoundUp(tiles, kLinearLocalSize),
                   kLinearLocalSize);
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
