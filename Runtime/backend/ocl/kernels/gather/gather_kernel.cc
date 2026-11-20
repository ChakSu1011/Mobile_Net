// 模块：OCL Gather Kernel 实现
//
// 职责：
//   校验视图/形状，编译 gather_embedding，绑定 ids/weight/out 与形状参数，
//   下发一维 NDRange。
//
// 边界：
//   P4.2 基础实现；不做 half8 之外的进一步调优；越界 id 由上层保证。
//
// 依赖：
//   Runtime/backend/ocl/kernels/gather/gather_kernel.h、
//   Runtime/backend/kernel_context.h、Runtime/common/data_types.h、
//   C++17 标准库。

#include "Runtime/backend/ocl/kernels/gather/gather_kernel.h"

#include <limits>
#include <memory>
#include <string>

#include "Runtime/common/data_types.h"

namespace mobilenet {
namespace ocl {
namespace {

// workgroup 大小；hidden 较大时每线程循环。
constexpr Usize kGatherLocalSize = 128;

}  // namespace

GatherOclKernel::GatherOclKernel(OclContext* context,
                                 OclProgramCache* programs,
                                 std::string kernel_key)
    : OclKernel(context, programs, std::move(kernel_key)) {}

Status GatherOclKernel::Prepare(
    const runtime_graph::KernelContext& context) {
  if (context.input_count < 2 || context.output_count < 1 ||
      context.inputs == nullptr || context.outputs == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "GatherOclKernel: insufficient tensors");
  }
  Status status = ValidateView(context.inputs[0], DataType::kInt64, "ids");
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
  const runtime_graph::Shape& ids_shape = context.inputs[0].shape;
  const runtime_graph::Shape& weight_shape = context.inputs[1].shape;
  const runtime_graph::Shape& out_shape = context.outputs[0].shape;
  if (ids_shape.max_dim != 2 || weight_shape.max_dim != 2 ||
      out_shape.max_dim != 3) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "GatherOclKernel: unexpected tensor rank");
  }
  const I64 tokens = ids_shape.GetShapeSize();
  const I64 vocab = weight_shape.dims[0];
  const I64 hidden = weight_shape.dims[1];
  if (tokens <= 0 || vocab <= 0 || hidden <= 0 ||
      hidden > std::numeric_limits<I32>::max()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "GatherOclKernel: invalid shape");
  }
  return Compile("gather_embedding", "-cl-mad-enable");
}

Status GatherOclKernel::ExecuteImpl(
    const runtime_graph::KernelContext& context) {
  if (context.input_count < 2 || context.output_count < 1) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "GatherOclKernel: insufficient tensors");
  }
  const runtime_graph::TensorView& ids = context.inputs[0];
  const runtime_graph::TensorView& weight = context.inputs[1];
  const runtime_graph::TensorView& out = context.outputs[0];
  const I64 tokens = ids.shape.GetShapeSize();
  const I64 vocab = weight.shape.dims[0];
  const I64 hidden = weight.shape.dims[1];
  if (tokens <= 0 || vocab <= 0 || hidden <= 0 ||
      tokens > std::numeric_limits<I32>::max() ||
      vocab > std::numeric_limits<I32>::max()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "GatherOclKernel: invalid shape");
  }

  I32 ids_offset = 0;
  I32 weight_offset = 0;
  I32 out_offset = 0;
  Status status = ElementOffset(ids, DataType::kInt64, &ids_offset);
  if (!status.ok()) {
    return status;
  }
  status = ElementOffset(weight, DataType::kFloat16, &weight_offset);
  if (!status.ok()) {
    return status;
  }
  status = ElementOffset(out, DataType::kFloat16, &out_offset);
  if (!status.ok()) {
    return status;
  }

  status = SetMemArg(0, ids);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(1, ids_offset);
  if (!status.ok()) {
    return status;
  }
  status = SetMemArg(2, weight);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(3, weight_offset);
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
  // 生产执行时只处理本步有效 token，padding 行不参与 embedding 查表。
  const I64 valid_tokens = EffectiveRows(context, tokens);
  status = SetIntArg(6, static_cast<I32>(valid_tokens));
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(7, static_cast<I32>(vocab));
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(8, static_cast<I32>(hidden));
  if (!status.ok()) {
    return status;
  }
  return Enqueue1D(static_cast<Usize>(valid_tokens) * kGatherLocalSize,
                   kGatherLocalSize);
}

Status CreateGatherOclKernel(OclContext* context,
                             OclProgramCache* programs,
                             const std::string& kernel_key,
                             std::unique_ptr<Kernel>* kernel) {
  if (kernel == nullptr || context == nullptr || programs == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "CreateGatherOclKernel: kernel/context/programs must not be null");
  }
  *kernel = std::make_unique<GatherOclKernel>(context, programs, kernel_key);
  return Status::Ok();
}

}  // namespace ocl
}  // namespace mobilenet
