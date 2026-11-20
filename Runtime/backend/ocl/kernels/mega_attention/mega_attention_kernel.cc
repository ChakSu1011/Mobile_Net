// 模块：OCL MegaAttention Kernel 实现
//
// 职责：
//   校验 Q/K/V/out 与 head 配置，读取 causal/scale，编译并下发
//   mega_attention_causal。
//
// 边界：
//   P4.3 基线：一 query 一 work-item；D ≤ 128；不物化 score 矩阵。
//
// 依赖：
//   Runtime/backend/ocl/kernels/mega_attention/mega_attention_kernel.h、
//   Runtime/backend/ops/attr_utils.h、Runtime/common/data_types.h。

#include "Runtime/backend/ocl/kernels/mega_attention/mega_attention_kernel.h"

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "Runtime/backend/ops/attr_utils.h"
#include "Runtime/common/data_types.h"

namespace mobilenet {
namespace ocl {
namespace {

constexpr I64 kAttentionMaxHeadDim = 128;

}  // namespace

MegaAttentionOclKernel::MegaAttentionOclKernel(
    OclContext* context, OclProgramCache* programs, std::string kernel_key)
    : OclKernel(context, programs, std::move(kernel_key)) {}

Status MegaAttentionOclKernel::Prepare(
    const runtime_graph::KernelContext& context) {
  if (context.input_count < 3 || context.output_count < 1 ||
      context.inputs == nullptr || context.outputs == nullptr ||
      context.attrs == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "MegaAttentionOclKernel: missing tensors/attrs");
  }
  Status status = ValidateView(context.inputs[0], DataType::kFloat16, "q");
  if (!status.ok()) {
    return status;
  }
  status = ValidateView(context.inputs[1], DataType::kFloat16, "k");
  if (!status.ok()) {
    return status;
  }
  status = ValidateView(context.inputs[2], DataType::kFloat16, "v");
  if (!status.ok()) {
    return status;
  }
  status = ValidateView(context.outputs[0], DataType::kFloat16, "out");
  if (!status.ok()) {
    return status;
  }
  const runtime_graph::Shape& q_shape = context.inputs[0].shape;
  const runtime_graph::Shape& k_shape = context.inputs[1].shape;
  const runtime_graph::Shape& v_shape = context.inputs[2].shape;
  if (q_shape.max_dim != 3 || k_shape.max_dim != 3 ||
      v_shape.max_dim != 3) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "MegaAttentionOclKernel: unexpected tensor rank");
  }
  I64 num_heads = 0;
  I64 num_kv_heads = 0;
  I64 head_dim = 0;
  bool causal = true;
  FP64 softmax_scale = 0.0;
  status = GetIntAttr(*context.attrs, "num_heads", &num_heads);
  if (!status.ok()) {
    return status;
  }
  status = GetIntAttr(*context.attrs, "num_kv_heads", &num_kv_heads);
  if (!status.ok()) {
    return status;
  }
  status = GetIntAttr(*context.attrs, "head_dim", &head_dim);
  if (!status.ok()) {
    return status;
  }
  status = GetBoolAttr(*context.attrs, "causal", &causal);
  if (!status.ok()) {
    return status;
  }
  status = GetFloatAttr(*context.attrs, "softmax_scale", &softmax_scale);
  if (!status.ok()) {
    return status;
  }
  const I64 batch = q_shape.dims[0];
  const I64 seq = q_shape.dims[1];
  if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
      head_dim > kAttentionMaxHeadDim || num_heads % num_kv_heads != 0 ||
      batch <= 0 || seq <= 0 ||
      q_shape.dims[2] != num_heads * head_dim ||
      k_shape.dims[2] != num_kv_heads * head_dim ||
      v_shape.dims[2] != num_kv_heads * head_dim ||
      k_shape.dims[0] != batch || k_shape.dims[1] != seq ||
      v_shape.dims[0] != batch || v_shape.dims[1] != seq ||
      context.outputs[0].shape.GetShapeSize() != q_shape.GetShapeSize() ||
      softmax_scale <= 0.0 ||
      batch > std::numeric_limits<I32>::max() ||
      seq > std::numeric_limits<I32>::max() ||
      num_heads > std::numeric_limits<I32>::max() ||
      num_kv_heads > std::numeric_limits<I32>::max() ||
      head_dim > std::numeric_limits<I32>::max()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "MegaAttentionOclKernel: invalid shape/attrs");
  }
  (void)causal;
  return Compile("mega_attention_causal", "-cl-mad-enable");
}

Status MegaAttentionOclKernel::ExecuteImpl(
    const runtime_graph::KernelContext& context) {
  if (context.input_count < 3 || context.output_count < 1 ||
      context.attrs == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "MegaAttentionOclKernel: missing tensors/attrs");
  }
  const runtime_graph::TensorView& q = context.inputs[0];
  const runtime_graph::TensorView& k = context.inputs[1];
  const runtime_graph::TensorView& v = context.inputs[2];
  const runtime_graph::TensorView& out = context.outputs[0];
  I64 num_heads = 0;
  I64 num_kv_heads = 0;
  I64 head_dim = 0;
  bool causal = true;
  FP64 softmax_scale = 0.0;
  Status status = GetIntAttr(*context.attrs, "num_heads", &num_heads);
  if (!status.ok()) {
    return status;
  }
  status = GetIntAttr(*context.attrs, "num_kv_heads", &num_kv_heads);
  if (!status.ok()) {
    return status;
  }
  status = GetIntAttr(*context.attrs, "head_dim", &head_dim);
  if (!status.ok()) {
    return status;
  }
  status = GetBoolAttr(*context.attrs, "causal", &causal);
  if (!status.ok()) {
    return status;
  }
  status = GetFloatAttr(*context.attrs, "softmax_scale", &softmax_scale);
  if (!status.ok()) {
    return status;
  }
  const I64 batch = q.shape.dims[0];
  const I64 seq = q.shape.dims[1];
  if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
      head_dim > kAttentionMaxHeadDim || num_heads % num_kv_heads != 0 ||
      batch <= 0 || seq <= 0 ||
      batch > std::numeric_limits<I32>::max() ||
      seq > std::numeric_limits<I32>::max() ||
      num_heads > std::numeric_limits<I32>::max() ||
      num_kv_heads > std::numeric_limits<I32>::max() ||
      head_dim > std::numeric_limits<I32>::max()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "MegaAttentionOclKernel: invalid shape/attrs");
  }

  I32 q_offset = 0;
  I32 k_offset = 0;
  I32 v_offset = 0;
  I32 out_offset = 0;
  status = ElementOffset(q, DataType::kFloat16, &q_offset);
  if (!status.ok()) {
    return status;
  }
  status = ElementOffset(k, DataType::kFloat16, &k_offset);
  if (!status.ok()) {
    return status;
  }
  status = ElementOffset(v, DataType::kFloat16, &v_offset);
  if (!status.ok()) {
    return status;
  }
  status = ElementOffset(out, DataType::kFloat16, &out_offset);
  if (!status.ok()) {
    return status;
  }

  status = SetMemArg(0, q);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(1, q_offset);
  if (!status.ok()) {
    return status;
  }
  status = SetMemArg(2, k);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(3, k_offset);
  if (!status.ok()) {
    return status;
  }
  status = SetMemArg(4, v);
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(5, v_offset);
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
  status = SetIntArg(8, static_cast<I32>(batch));
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(9, static_cast<I32>(seq));
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(10, static_cast<I32>(num_heads));
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(11, static_cast<I32>(num_kv_heads));
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(12, static_cast<I32>(head_dim));
  if (!status.ok()) {
    return status;
  }
  status = SetFloatArg(13, static_cast<float>(softmax_scale));
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(14, causal ? 1 : 0);
  if (!status.ok()) {
    return status;
  }
  const Usize work = static_cast<Usize>(batch * num_heads * seq);
  return Enqueue1D(work, 1);
}

Status CreateMegaAttentionOclKernel(OclContext* context,
                                    OclProgramCache* programs,
                                    const std::string& kernel_key,
                                    std::unique_ptr<Kernel>* kernel) {
  if (kernel == nullptr || context == nullptr || programs == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "CreateMegaAttentionOclKernel: null argument");
  }
  *kernel = std::make_unique<MegaAttentionOclKernel>(context, programs,
                                                     kernel_key);
  return Status::Ok();
}

}  // namespace ocl
}  // namespace mobilenet
