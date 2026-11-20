// 模块：OCL Add Kernel 实现
//
// 职责：
//   校验三视图同形，编译 add_half，按元素/偏移对齐选择 half8 或标量路径，
//   绑定参数并下发。
//
// 边界：
//   P4.2 基线；不做广播；性能调优在后续阶段。
//
// 依赖：
//   Runtime/backend/ocl/kernels/add/add_kernel.h、
//   Runtime/common/data_types.h、C++17 标准库。

#include "Runtime/backend/ocl/kernels/add/add_kernel.h"

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "Runtime/common/data_types.h"

namespace mobilenet {
namespace ocl {
namespace {

constexpr Usize kAddLocalSize = 128;

// 计算向上取整到 local 倍数的 global size。
Usize RoundUp(Usize value, Usize local) {
  return ((value + local - 1) / local) * local;
}

}  // namespace

AddOclKernel::AddOclKernel(OclContext* context, OclProgramCache* programs,
                           std::string kernel_key)
    : OclKernel(context, programs, std::move(kernel_key)) {}

Status AddOclKernel::Prepare(
    const runtime_graph::KernelContext& context) {
  if (context.input_count < 2 || context.output_count < 1 ||
      context.inputs == nullptr || context.outputs == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "AddOclKernel: insufficient tensors");
  }
  Status status = ValidateView(context.inputs[0], DataType::kFloat16, "a");
  if (!status.ok()) {
    return status;
  }
  status = ValidateView(context.inputs[1], DataType::kFloat16, "b");
  if (!status.ok()) {
    return status;
  }
  status = ValidateView(context.outputs[0], DataType::kFloat16, "out");
  if (!status.ok()) {
    return status;
  }
  const I64 elements = context.inputs[0].shape.GetShapeSize();
  if (elements <= 0 || elements > std::numeric_limits<I32>::max() ||
      context.inputs[1].shape.GetShapeSize() != elements ||
      context.outputs[0].shape.GetShapeSize() != elements) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "AddOclKernel: invalid/shape-mismatched tensors");
  }
  return Compile("add_half", "-cl-mad-enable");
}

Status AddOclKernel::ExecuteImpl(
    const runtime_graph::KernelContext& context) {
  if (context.input_count < 2 || context.output_count < 1) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "AddOclKernel: insufficient tensors");
  }
  const runtime_graph::TensorView& a = context.inputs[0];
  const runtime_graph::TensorView& b = context.inputs[1];
  const runtime_graph::TensorView& out = context.outputs[0];
  const I64 elements = a.shape.GetShapeSize();
  if (elements <= 0 || elements > std::numeric_limits<I32>::max() ||
      b.shape.GetShapeSize() != elements ||
      out.shape.GetShapeSize() != elements) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "AddOclKernel: invalid/shape-mismatched tensors");
  }

  I32 a_offset = 0;
  I32 b_offset = 0;
  I32 out_offset = 0;
  Status status = ElementOffset(a, DataType::kFloat16, &a_offset);
  if (!status.ok()) {
    return status;
  }
  status = ElementOffset(b, DataType::kFloat16, &b_offset);
  if (!status.ok()) {
    return status;
  }
  status = ElementOffset(out, DataType::kFloat16, &out_offset);
  if (!status.ok()) {
    return status;
  }

  // 生产执行时只处理本步有效 token 对应的元素，跳过 padding 行。
  const I64 last_dim = a.shape.dims[a.shape.max_dim - 1];
  I64 valid_elements = elements;
  if (last_dim > 0 && elements % last_dim == 0) {
    valid_elements = EffectiveRows(context, elements / last_dim) * last_dim;
  }
  const bool vectorized = (valid_elements % 8 == 0) &&
                          (a_offset % 8 == 0) && (b_offset % 8 == 0) &&
                          (out_offset % 8 == 0);
  const Usize work_items =
      vectorized ? static_cast<Usize>(valid_elements / 8)
                 : static_cast<Usize>(valid_elements);
  const Usize global = RoundUp(work_items, kAddLocalSize);

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
  status = SetMemArg(4, out);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(5, out_offset);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(6, static_cast<I32>(valid_elements));
  if (!status.ok()) {
    return status;
  }
  return Enqueue1D(global, kAddLocalSize);
}

Status CreateAddOclKernel(OclContext* context, OclProgramCache* programs,
                          const std::string& kernel_key,
                          std::unique_ptr<Kernel>* kernel) {
  if (kernel == nullptr || context == nullptr || programs == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "CreateAddOclKernel: kernel/context/programs must not be null");
  }
  *kernel = std::make_unique<AddOclKernel>(context, programs, kernel_key);
  return Status::Ok();
}

}  // namespace ocl
}  // namespace mobilenet
