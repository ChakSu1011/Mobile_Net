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
  // 生产执行时只处理有效 token：seq 为有效长度，seq_stride 保持静态行
  // 跨度，二者分开避免把 packed 行距当作物理行距。
  const I64 valid_seq = EffectiveRows(context, seq);
  status = SetIntArg(8, static_cast<I32>(batch));
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(9, static_cast<I32>(valid_seq));
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(10, static_cast<I32>(seq));
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(11, static_cast<I32>(num_heads));
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(12, static_cast<I32>(num_kv_heads));
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(13, static_cast<I32>(head_dim));
  if (!status.ok()) {
    return status;
  }
  status = SetFloatArg(14, static_cast<float>(softmax_scale));
  if (!status.ok()) {
    return status;
  }
  status = SetIntArg(15, causal ? 1 : 0);
  if (!status.ok()) {
    return status;
  }
  const Usize work =
      static_cast<Usize>(batch * num_heads * valid_seq);
  return Enqueue1D(work, 1);
}

MegaAttentionPagedOclKernel::MegaAttentionPagedOclKernel(
    OclContext* context, OclProgramCache* programs, std::string kernel_key)
    : OclKernel(context, programs, std::move(kernel_key)) {}

MegaAttentionPagedOclKernel::~MegaAttentionPagedOclKernel() {
  ReleaseExtraKernel(&store_kernel_);
  ReleaseExtraKernel(&attention_kernel_);
}

Status MegaAttentionPagedOclKernel::Prepare(
    const runtime_graph::KernelContext& context) {
  if (context.input_count < 3 || context.output_count < 1 ||
      context.inputs == nullptr || context.outputs == nullptr ||
      context.attrs == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "MegaAttentionPagedOclKernel: missing tensors/attrs");
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
  if (context.attention == nullptr || context.attention->kv == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "MegaAttentionPagedOclKernel: missing attention/KV");
  }
  I64 num_heads = 0;
  I64 num_kv_heads = 0;
  I64 head_dim = 0;
  Bool causal = true;
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
  const runtime_graph::Shape& q_shape = context.inputs[0].shape;
  const runtime_graph::Shape& k_shape = context.inputs[1].shape;
  const runtime_graph::Shape& v_shape = context.inputs[2].shape;
  if (q_shape.max_dim != 3 || k_shape.max_dim != 3 ||
      v_shape.max_dim != 3 || num_heads <= 0 || num_kv_heads <= 0 ||
      head_dim <= 0 || head_dim > kAttentionMaxHeadDim ||
      num_heads % num_kv_heads != 0 ||
      q_shape.dims[2] != num_heads * head_dim ||
      k_shape.dims[2] != num_kv_heads * head_dim ||
      v_shape.dims[2] != num_kv_heads * head_dim ||
      k_shape.dims[0] != q_shape.dims[0] ||
      k_shape.dims[1] != q_shape.dims[1] ||
      v_shape.dims[0] != q_shape.dims[0] ||
      v_shape.dims[1] != q_shape.dims[1] ||
      context.outputs[0].shape.GetShapeSize() !=
          q_shape.GetShapeSize() ||
      softmax_scale <= 0.0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "MegaAttentionPagedOclKernel: invalid shape/attrs");
  }
  status = CompileExtra("paged_kv_store", "-cl-mad-enable",
                        &store_kernel_);
  if (!status.ok()) {
    return status;
  }
  status = CompileExtra("mega_attention_paged", "-cl-mad-enable",
                        &attention_kernel_);
  if (!status.ok()) {
    return status;
  }
  (void)causal;
  return Status::Ok();
}

Status MegaAttentionPagedOclKernel::ExecuteImpl(
    const runtime_graph::KernelContext& context) {
  if (store_kernel_ == nullptr || attention_kernel_ == nullptr ||
      context.attention == nullptr || context.attention->kv == nullptr) {
    return MakeStatus(ErrorCode::kStateError,
                      "MegaAttentionPagedOclKernel: kernel/attention missing");
  }
  const runtime_graph::AttentionContext& attention = *context.attention;
  const runtime::KvCacheView& kv = *attention.kv;
  if (attention.layer_index < 0 ||
      static_cast<Usize>(attention.layer_index) >= kv.layers.size()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "MegaAttentionPagedOclKernel: layer index out of range");
  }
  const runtime::KvLayerView& layer =
      kv.layers[static_cast<Usize>(attention.layer_index)];
  if (layer.dtype != DataType::kFloat16 || layer.k_buffer == nullptr ||
      layer.v_buffer == nullptr || layer.block_stride <= 0 ||
      kv.block_size <= 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "MegaAttentionPagedOclKernel: invalid KV layer view");
  }
  const I64 expected_stride =
      static_cast<I64>(kv.block_size) * layer.num_kv_heads *
      layer.head_dim * static_cast<I64>(sizeof(FP16));
  if (layer.block_stride != expected_stride) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "MegaAttentionPagedOclKernel: KV block stride has padding; "
        "NHD contiguous layout is required");
  }
  const runtime_graph::AttentionMetadataView& meta = attention.meta;
  Status status = ValidateView(meta.seq_lens, DataType::kInt32, "seq_lens");
  if (!status.ok()) return status;
  status = ValidateView(meta.query_lens, DataType::kInt32, "query_lens");
  if (!status.ok()) return status;
  status = ValidateView(meta.context_lens, DataType::kInt32, "context_lens");
  if (!status.ok()) return status;
  status = ValidateView(meta.request_indices, DataType::kInt32,
                        "request_indices");
  if (!status.ok()) return status;
  status = ValidateView(meta.positions, DataType::kInt32, "positions");
  if (!status.ok()) return status;
  status = ValidateView(meta.slots, DataType::kInt64, "slots");
  if (!status.ok()) return status;
  status = ValidateView(meta.block_table, DataType::kInt32, "block_table");
  if (!status.ok()) return status;

  const I64 batch = attention.batch;
  const I64 token_count = attention.token_count;
  const I64 static_seq = context.inputs[0].shape.dims[1];
  if (batch <= 0 || token_count <= 0 || token_count > static_seq ||
      meta.block_table_stride <= 0 ||
      meta.seq_lens.shape.GetShapeSize() < batch ||
      meta.query_lens.shape.GetShapeSize() < batch ||
      meta.context_lens.shape.GetShapeSize() < batch ||
      meta.request_indices.shape.GetShapeSize() < token_count ||
      meta.positions.shape.GetShapeSize() < token_count ||
      meta.slots.shape.GetShapeSize() < token_count ||
      meta.block_table.shape.GetShapeSize() <
          batch * meta.block_table_stride) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "MegaAttentionPagedOclKernel: metadata size mismatch");
  }

  I64 num_heads = 0;
  I64 num_kv_heads = 0;
  I64 head_dim = 0;
  Bool causal = true;
  FP64 softmax_scale = 0.0;
  status = GetIntAttr(*context.attrs, "num_heads", &num_heads);
  if (!status.ok()) return status;
  status = GetIntAttr(*context.attrs, "num_kv_heads", &num_kv_heads);
  if (!status.ok()) return status;
  status = GetIntAttr(*context.attrs, "head_dim", &head_dim);
  if (!status.ok()) return status;
  status = GetBoolAttr(*context.attrs, "causal", &causal);
  if (!status.ok()) return status;
  status = GetFloatAttr(*context.attrs, "softmax_scale", &softmax_scale);
  if (!status.ok()) return status;
  if (layer.num_kv_heads != num_kv_heads || layer.head_dim != head_dim) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "MegaAttentionPagedOclKernel: KV view heads/dim do not match attrs");
  }

  I32 q_offset = 0;
  I32 k_offset = 0;
  I32 v_offset = 0;
  I32 out_offset = 0;
  I32 seq_offset = 0;
  I32 query_offset = 0;
  I32 context_offset = 0;
  I32 request_offset = 0;
  I32 position_offset = 0;
  I32 slot_offset = 0;
  I32 block_table_offset = 0;
  status = ElementOffset(context.inputs[0], DataType::kFloat16, &q_offset);
  if (!status.ok()) return status;
  status = ElementOffset(context.inputs[1], DataType::kFloat16, &k_offset);
  if (!status.ok()) return status;
  status = ElementOffset(context.inputs[2], DataType::kFloat16, &v_offset);
  if (!status.ok()) return status;
  status = ElementOffset(context.outputs[0], DataType::kFloat16, &out_offset);
  if (!status.ok()) return status;
  status = ElementOffset(meta.seq_lens, DataType::kInt32, &seq_offset);
  if (!status.ok()) return status;
  status = ElementOffset(meta.query_lens, DataType::kInt32, &query_offset);
  if (!status.ok()) return status;
  status = ElementOffset(meta.context_lens, DataType::kInt32,
                         &context_offset);
  if (!status.ok()) return status;
  status = ElementOffset(meta.request_indices, DataType::kInt32,
                         &request_offset);
  if (!status.ok()) return status;
  status = ElementOffset(meta.positions, DataType::kInt32, &position_offset);
  if (!status.ok()) return status;
  status = ElementOffset(meta.slots, DataType::kInt64, &slot_offset);
  if (!status.ok()) return status;
  status = ElementOffset(meta.block_table, DataType::kInt32,
                         &block_table_offset);
  if (!status.ok()) return status;

  runtime_graph::TensorView k_pool_view;
  k_pool_view.data = layer.k_buffer;
  k_pool_view.offset = 0;
  k_pool_view.bytes = static_cast<U64>(layer.num_blocks) *
                      static_cast<U64>(layer.block_stride);
  k_pool_view.dtype = DataType::kFloat16;
  k_pool_view.layout = runtime_graph::Layout::kRowMajor;
  k_pool_view.device = runtime_graph::DeviceType::kDevice;
  k_pool_view.shape.max_dim = 2;
  k_pool_view.shape.dims[0] = layer.num_blocks * kv.block_size;
  k_pool_view.shape.dims[1] = layer.num_kv_heads * layer.head_dim;
  k_pool_view.shape.pitch[0] = k_pool_view.shape.dims[0];
  k_pool_view.shape.pitch[1] = k_pool_view.shape.dims[1];
  runtime_graph::TensorView v_pool_view = k_pool_view;
  v_pool_view.data = layer.v_buffer;

  // 第一段：写入本步 K/V。
  status = SetMemArg(store_kernel_, 0, context.inputs[1]);
  if (!status.ok()) return status;
  status = SetIntArg(store_kernel_, 1, k_offset);
  if (!status.ok()) return status;
  status = SetMemArg(store_kernel_, 2, context.inputs[2]);
  if (!status.ok()) return status;
  status = SetIntArg(store_kernel_, 3, v_offset);
  if (!status.ok()) return status;
  status = SetMemArg(store_kernel_, 4, k_pool_view);
  if (!status.ok()) return status;
  status = SetMemArg(store_kernel_, 5, v_pool_view);
  if (!status.ok()) return status;
  status = SetMemArg(store_kernel_, 6, meta.slots);
  if (!status.ok()) return status;
  status = SetIntArg(store_kernel_, 7, slot_offset);
  if (!status.ok()) return status;
  status = SetIntArg(store_kernel_, 8, static_cast<I32>(token_count));
  if (!status.ok()) return status;
  status = SetIntArg(store_kernel_, 9, static_cast<I32>(num_kv_heads));
  if (!status.ok()) return status;
  status = SetIntArg(store_kernel_, 10, static_cast<I32>(head_dim));
  if (!status.ok()) return status;
  const Usize store_work =
      static_cast<Usize>(token_count) * static_cast<Usize>(num_kv_heads);
  status = Enqueue1D(store_kernel_, store_work, 1);
  if (!status.ok()) return status;

  // 第二段：按块表读取历史 KV。
  status = SetMemArg(attention_kernel_, 0, context.inputs[0]);
  if (!status.ok()) return status;
  status = SetIntArg(attention_kernel_, 1, q_offset);
  if (!status.ok()) return status;
  status = SetMemArg(attention_kernel_, 2, context.outputs[0]);
  if (!status.ok()) return status;
  status = SetIntArg(attention_kernel_, 3, out_offset);
  if (!status.ok()) return status;
  status = SetMemArg(attention_kernel_, 4, k_pool_view);
  if (!status.ok()) return status;
  status = SetMemArg(attention_kernel_, 5, v_pool_view);
  if (!status.ok()) return status;
  status = SetMemArg(attention_kernel_, 6, meta.seq_lens);
  if (!status.ok()) return status;
  status = SetIntArg(attention_kernel_, 7, seq_offset);
  if (!status.ok()) return status;
  status = SetMemArg(attention_kernel_, 8, meta.query_lens);
  if (!status.ok()) return status;
  status = SetIntArg(attention_kernel_, 9, query_offset);
  if (!status.ok()) return status;
  status = SetMemArg(attention_kernel_, 10, meta.context_lens);
  if (!status.ok()) return status;
  status = SetIntArg(attention_kernel_, 11, context_offset);
  if (!status.ok()) return status;
  status = SetMemArg(attention_kernel_, 12, meta.request_indices);
  if (!status.ok()) return status;
  status = SetIntArg(attention_kernel_, 13, request_offset);
  if (!status.ok()) return status;
  status = SetMemArg(attention_kernel_, 14, meta.positions);
  if (!status.ok()) return status;
  status = SetIntArg(attention_kernel_, 15, position_offset);
  if (!status.ok()) return status;
  status = SetMemArg(attention_kernel_, 16, meta.block_table);
  if (!status.ok()) return status;
  status = SetIntArg(attention_kernel_, 17, block_table_offset);
  if (!status.ok()) return status;
  status = SetIntArg(attention_kernel_, 18, static_cast<I32>(batch));
  if (!status.ok()) return status;
  status = SetIntArg(attention_kernel_, 19, static_cast<I32>(token_count));
  if (!status.ok()) return status;
  status = SetIntArg(attention_kernel_, 20, static_cast<I32>(static_seq));
  if (!status.ok()) return status;
  status = SetIntArg(attention_kernel_, 21, meta.block_table_stride);
  if (!status.ok()) return status;
  status = SetIntArg(attention_kernel_, 22, static_cast<I32>(num_heads));
  if (!status.ok()) return status;
  status = SetIntArg(attention_kernel_, 23, static_cast<I32>(num_kv_heads));
  if (!status.ok()) return status;
  status = SetIntArg(attention_kernel_, 24, static_cast<I32>(head_dim));
  if (!status.ok()) return status;
  status = SetIntArg(attention_kernel_, 25, kv.block_size);
  if (!status.ok()) return status;
  status = SetFloatArg(attention_kernel_, 26,
                       static_cast<float>(softmax_scale));
  if (!status.ok()) return status;
  status = SetIntArg(attention_kernel_, 27, causal ? 1 : 0);
  if (!status.ok()) return status;
  const Usize attention_work =
      static_cast<Usize>(token_count) * static_cast<Usize>(num_heads);
  return Enqueue1D(attention_kernel_, attention_work, 1);
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

Status CreateMegaAttentionPagedOclKernel(
    OclContext* context, OclProgramCache* programs,
    const std::string& kernel_key, std::unique_ptr<Kernel>* kernel) {
  if (kernel == nullptr || context == nullptr || programs == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "CreateMegaAttentionPagedOclKernel: null argument");
  }
  *kernel = std::make_unique<MegaAttentionPagedOclKernel>(
      context, programs, kernel_key);
  return Status::Ok();
}

}  // namespace ocl
}  // namespace mobilenet
