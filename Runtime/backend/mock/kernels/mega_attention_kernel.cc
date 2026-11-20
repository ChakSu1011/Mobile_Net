// 模块：Mock MegaAttention Kernel
//
// 职责：
//   实现 causal GQA attention 的 CPU 参考计算：
//   out = softmax(Q·K^T * scale + causal_mask) · V。
//
// 边界：
//   - 只支持 merge_heads=true、has_output_projection=false、
//     use_kv_cache=false；
//   - GQA 通过 num_heads/num_kv_heads 的整除关系复用 K/V；
//   - FP32 online softmax，FP16 写回；
//   - 不做分块、向量化或并行。
//
// 依赖：
//   Runtime/backend/mock/kernels/kernel_common.h、
//   Runtime/backend/ops/attr_utils.h、mock_kernel_creators.h。

#include "Runtime/backend/mock/kernels/mock_kernel_creators.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "Runtime/backend/mock/half_utils.h"
#include "Runtime/backend/mock/kernels/kernel_common.h"
#include "Runtime/backend/ops/attr_utils.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace mock {
namespace {

// CPU 参考 GQA Attention Kernel。
class MegaAttentionKernel final : public Kernel {
 public:
  Status Execute(const runtime_graph::KernelContext& context) override {
    Status status = RequireInputCount(context, 3);
    if (!status.ok()) {
      return status;
    }
    status = RequireOutputCount(context, 1);
    if (!status.ok()) {
      return status;
    }
    status = RequireAttrs(context);
    if (!status.ok()) {
      return status;
    }

    const std::vector<I64> query = Dims(context.inputs[0].shape);
    const std::vector<I64> key = Dims(context.inputs[1].shape);
    const std::vector<I64> value = Dims(context.inputs[2].shape);
    const std::vector<I64> output = Dims(context.outputs[0].shape);
    if (query.size() != 3 || key.size() != 3 || value.size() != 3 ||
        output != query) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "MegaAttentionKernel: unexpected tensor shape");
    }
    const I64 batch = query[0];
    const I64 sequence = query[1];
    if (batch < 0 || sequence <= 0) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "MegaAttentionKernel: invalid batch/sequence");
    }
    if (key[0] != batch || key[1] != sequence ||
        value[0] != batch || value[1] != sequence) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "MegaAttentionKernel: K/V prefix mismatch");
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
    if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
        num_heads % num_kv_heads != 0) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "MegaAttentionKernel: invalid head configuration");
    }
    if (query[2] != num_heads * head_dim ||
        key[2] != num_kv_heads * head_dim ||
        value[2] != num_kv_heads * head_dim) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "MegaAttentionKernel: head width mismatch");
    }
    if (softmax_scale <= 0.0) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "MegaAttentionKernel: invalid softmax_scale");
    }

    const I64 group = num_heads / num_kv_heads;
    const FP32 scale = static_cast<FP32>(softmax_scale);
    std::vector<FP32> accumulator(static_cast<Usize>(head_dim), 0.0f);
    for (I64 b = 0; b < batch; ++b) {
      for (I64 i = 0; i < sequence; ++i) {
        const I64 query_base = (b * sequence + i) * query[2];
        for (I64 head = 0; head < num_heads; ++head) {
          const I64 kv_head = head / group;
          const I64 query_head_base = query_base + head * head_dim;
          const I64 kv_base = (b * sequence) * key[2] +
                                  kv_head * head_dim;
          FP32 max_score = -std::numeric_limits<FP32>::infinity();
          const I64 last_key = causal ? i : sequence - 1;
          for (I64 j = 0; j <= last_key; ++j) {
            FP32 dot = 0.0f;
            const I64 key_base = kv_base + j * key[2];
            for (I64 d = 0; d < head_dim; ++d) {
              dot += ReadHalf(context.inputs[0], query_head_base + d) *
                     ReadHalf(context.inputs[1], key_base + d);
            }
            const FP32 score = dot * scale;
            if (score > max_score) {
              max_score = score;
            }
          }

          FP32 sum = 0.0f;
          std::fill(accumulator.begin(), accumulator.end(), 0.0f);
          for (I64 j = 0; j <= last_key; ++j) {
            FP32 dot = 0.0f;
            const I64 key_base = kv_base + j * key[2];
            for (I64 d = 0; d < head_dim; ++d) {
              dot += ReadHalf(context.inputs[0], query_head_base + d) *
                     ReadHalf(context.inputs[1], key_base + d);
            }
            const FP32 probability = std::exp(dot * scale - max_score);
            sum += probability;
            for (I64 d = 0; d < head_dim; ++d) {
              const I64 value_index =
                  (b * sequence + j) * value[2] + kv_head * head_dim + d;
              accumulator[static_cast<Usize>(d)] +=
                  probability * ReadHalf(context.inputs[2], value_index);
            }
          }
          if (!(sum > 0.0f)) {
            return MakeStatus(ErrorCode::kKernelError,
                              "MegaAttentionKernel: softmax sum is zero");
          }
          const FP32 inverse = 1.0f / sum;
          const I64 output_base = query_base + head * head_dim;
          for (I64 d = 0; d < head_dim; ++d) {
            WriteHalf(context.outputs[0], output_base + d,
                      accumulator[static_cast<Usize>(d)] * inverse);
          }
        }
      }
    }
    return Status::Ok();
  }
};

// Paged GQA Attention 的 CPU 参考实现。
//
// 功能：
//   按 AttentionMetadataView 与 KvLayerView 完成两段计算：
//   1. 把本步 K/V 写入 KV 池的 slot；
//   2. 每个 query token/head 按块表读取历史 K/V，执行 causal online
//      softmax。
//
// 边界：
//   - 只支持 FP16 K/V 池、NHD 布局、batch=1 的静态图；
//   - 有效 token 由 attention.token_count 截断；
//   - 不分配设备内存，不修改元数据。
class MegaAttentionPagedKernel final : public Kernel {
 public:
  Status Execute(const runtime_graph::KernelContext& context) override {
    Status status = RequireInputCount(context, 3);
    if (!status.ok()) {
      return status;
    }
    status = RequireOutputCount(context, 1);
    if (!status.ok()) {
      return status;
    }
    status = RequireAttrs(context);
    if (!status.ok()) {
      return status;
    }
    if (context.attention == nullptr || context.attention->kv == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "MegaAttentionPagedKernel: missing attention/KV");
    }
    const runtime_graph::AttentionContext& attention = *context.attention;
    const runtime::KvCacheView& kv = *attention.kv;
    if (attention.layer_index < 0 ||
        static_cast<Usize>(attention.layer_index) >= kv.layers.size()) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "MegaAttentionPagedKernel: layer index out of range");
    }
    const runtime::KvLayerView& layer =
        kv.layers[static_cast<Usize>(attention.layer_index)];
    if (layer.dtype != DataType::kFloat16 || layer.k_buffer == nullptr ||
        layer.v_buffer == nullptr) {
      return MakeStatus(ErrorCode::kUnsupportedDtype,
                        "MegaAttentionPagedKernel: KV must be FP16 buffers");
    }
    if (kv.block_size <= 0 || layer.block_stride <= 0 ||
        layer.block_stride % kv.block_size != 0 ||
        (layer.block_stride / kv.block_size) % sizeof(FP16) != 0) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "MegaAttentionPagedKernel: invalid KV block stride");
    }

    const auto* k_pool_buffer =
        static_cast<const MockBuffer*>(layer.k_buffer);
    const auto* v_pool_buffer =
        static_cast<const MockBuffer*>(layer.v_buffer);
    if (k_pool_buffer == nullptr || v_pool_buffer == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "MegaAttentionPagedKernel: null KV buffer");
    }
    auto* k_pool = static_cast<FP16*>(k_pool_buffer->Data());
    auto* v_pool = static_cast<FP16*>(v_pool_buffer->Data());
    const FP16* k_input = HalfData(context.inputs[1]);
    const FP16* v_input = HalfData(context.inputs[2]);
    if (k_pool == nullptr || v_pool == nullptr || k_input == nullptr ||
        v_input == nullptr) {
      return MakeStatus(ErrorCode::kInternal,
                        "MegaAttentionPagedKernel: KV data is null");
    }

    const std::vector<I64> query = Dims(context.inputs[0].shape);
    const std::vector<I64> key = Dims(context.inputs[1].shape);
    const std::vector<I64> value = Dims(context.inputs[2].shape);
    const std::vector<I64> output = Dims(context.outputs[0].shape);
    if (query.size() != 3 || key.size() != 3 || value.size() != 3 ||
        output != query) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "MegaAttentionPagedKernel: unexpected tensor shape");
    }
    const I64 sequence = query[1];
    if (sequence <= 0 || key[1] != sequence || value[1] != sequence) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "MegaAttentionPagedKernel: invalid sequence");
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
    if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
        num_heads % num_kv_heads != 0 ||
        query[2] != num_heads * head_dim ||
        key[2] != num_kv_heads * head_dim ||
        value[2] != num_kv_heads * head_dim ||
        softmax_scale <= 0.0) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "MegaAttentionPagedKernel: invalid head config");
    }

    const runtime_graph::AttentionMetadataView& meta = attention.meta;
    const I32* seq_lens = Int32Data(meta.seq_lens);
    const I32* query_lens = Int32Data(meta.query_lens);
    const I32* context_lens = Int32Data(meta.context_lens);
    const I32* request_indices = Int32Data(meta.request_indices);
    const I32* positions = Int32Data(meta.positions);
    const I64* slots = Int64Data(meta.slots);
    const I32* block_table = Int32Data(meta.block_table);
    if (seq_lens == nullptr || query_lens == nullptr ||
        context_lens == nullptr || request_indices == nullptr ||
        positions == nullptr || slots == nullptr ||
        block_table == nullptr || meta.block_table_stride <= 0) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "MegaAttentionPagedKernel: missing metadata");
    }
    const I64 batch = attention.batch;
    const I64 token_count = attention.token_count;
    if (batch <= 0 || token_count <= 0 || token_count > sequence ||
        meta.seq_lens.shape.GetShapeSize() < batch ||
        meta.query_lens.shape.GetShapeSize() < batch ||
        meta.context_lens.shape.GetShapeSize() < batch ||
        meta.request_indices.shape.GetShapeSize() < token_count ||
        meta.positions.shape.GetShapeSize() < token_count ||
        meta.slots.shape.GetShapeSize() < token_count ||
        meta.block_table.shape.GetShapeSize() <
            batch * meta.block_table_stride) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "MegaAttentionPagedKernel: metadata size mismatch");
    }

    const I64 token_stride_bytes = layer.block_stride / kv.block_size;
    const I64 token_stride_elems = token_stride_bytes / sizeof(FP16);
    const I64 capacity_slots =
        layer.num_blocks * static_cast<I64>(kv.block_size);
    const I64 group = num_heads / num_kv_heads;
    const FP32 scale = static_cast<FP32>(softmax_scale);
    std::vector<FP32> accumulator(static_cast<Usize>(head_dim), 0.0f);

    for (I64 t = 0; t < token_count; ++t) {
      const I32 request = request_indices[static_cast<Usize>(t)];
      if (request < 0 || request >= batch) {
        return MakeStatus(ErrorCode::kInvalidArgument,
                          "MegaAttentionPagedKernel: request index invalid");
      }
      const I32 query_position = positions[static_cast<Usize>(t)];
      const I32 seq_len = seq_lens[static_cast<Usize>(request)];
      const I32 query_len = query_lens[static_cast<Usize>(request)];
      const I32 context_len = context_lens[static_cast<Usize>(request)];
      if (seq_len != context_len + query_len || seq_len <= 0 ||
          query_position < 0 || query_position >= seq_len) {
        return MakeStatus(ErrorCode::kInvalidArgument,
                          "MegaAttentionPagedKernel: invalid sequence meta");
      }

      const I64 slot = slots[static_cast<Usize>(t)];
      if (slot < 0 || slot >= capacity_slots) {
        return MakeStatus(ErrorCode::kInvalidArgument,
                          "MegaAttentionPagedKernel: slot out of range");
      }
      for (I64 kh = 0; kh < num_kv_heads; ++kh) {
        const I64 input_base = (t * num_kv_heads + kh) * head_dim;
        const I64 pool_base = slot * token_stride_elems +
                              kh * head_dim;
        for (I64 d = 0; d < head_dim; ++d) {
          // K/V 输入是 half；直接按位拷贝，不做数值转换。
          k_pool[pool_base + d] = k_input[input_base + d];
          v_pool[pool_base + d] = v_input[input_base + d];
        }
      }
      for (I64 head = 0; head < num_heads; ++head) {
        const I64 kv_head = head / group;
        const I64 query_base = (t * num_heads + head) * head_dim;
        FP32 max_score = -std::numeric_limits<FP32>::infinity();
        FP32 sum = 0.0f;
        std::fill(accumulator.begin(), accumulator.end(), 0.0f);
        for (I32 kv_pos = 0; kv_pos < seq_len; ++kv_pos) {
          if (causal && kv_pos > query_position) {
            break;
          }
          const I32 block_index = kv_pos / kv.block_size;
          const I32 block_offset = kv_pos % kv.block_size;
          const I64 table_index =
              static_cast<I64>(request) * meta.block_table_stride +
              block_index;
          if (table_index >=
              meta.block_table.shape.GetShapeSize()) {
            return MakeStatus(ErrorCode::kInvalidArgument,
                              "MegaAttentionPagedKernel: block table overflow");
          }
          const I32 block = block_table[table_index];
          if (block < 0) {
            return MakeStatus(ErrorCode::kInvalidArgument,
                              "MegaAttentionPagedKernel: negative block id");
          }
          const I64 pool_slot =
              static_cast<I64>(block) * kv.block_size + block_offset;
          if (pool_slot >= capacity_slots) {
            return MakeStatus(ErrorCode::kInvalidArgument,
                              "MegaAttentionPagedKernel: block out of range");
          }
          const I64 kv_base =
              pool_slot * token_stride_elems + kv_head * head_dim;
          FP32 dot = 0.0f;
          for (I64 d = 0; d < head_dim; ++d) {
            dot += ReadHalf(context.inputs[0], query_base + d) *
                   HalfToFloat(k_pool[kv_base + d]);
          }
          const FP32 score = dot * scale;
          if (score > max_score) {
            const FP32 alpha =
                max_score == -std::numeric_limits<FP32>::infinity()
                    ? 0.0f
                    : std::exp(max_score - score);
            sum *= alpha;
            for (I64 d = 0; d < head_dim; ++d) {
              accumulator[static_cast<Usize>(d)] *= alpha;
            }
            max_score = score;
            sum += 1.0f;
            for (I64 d = 0; d < head_dim; ++d) {
              accumulator[static_cast<Usize>(d)] +=
                  HalfToFloat(v_pool[kv_base + d]);
            }
          } else {
            const FP32 probability = std::exp(score - max_score);
            sum += probability;
            for (I64 d = 0; d < head_dim; ++d) {
              accumulator[static_cast<Usize>(d)] +=
                  probability * HalfToFloat(v_pool[kv_base + d]);
            }
          }
        }
        if (!(sum > 0.0f)) {
          return MakeStatus(ErrorCode::kKernelError,
                            "MegaAttentionPagedKernel: softmax sum is zero");
        }
        const FP32 inverse = 1.0f / sum;
        for (I64 d = 0; d < head_dim; ++d) {
          WriteHalf(context.outputs[0], query_base + d,
                    accumulator[static_cast<Usize>(d)] * inverse);
        }
      }
    }
    return Status::Ok();
  }
};

}  // namespace

Status CreateMegaAttentionKernel(std::unique_ptr<Kernel>* kernel) {
  if (kernel == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "CreateMegaAttentionKernel: kernel output must not be null");
  }
  *kernel = std::make_unique<MegaAttentionKernel>();
  return Status::Ok();
}

Status CreateMegaAttentionPagedKernel(std::unique_ptr<Kernel>* kernel) {
  if (kernel == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "CreateMegaAttentionPagedKernel: kernel output must not be null");
  }
  *kernel = std::make_unique<MegaAttentionPagedKernel>();
  return Status::Ok();
}

}  // namespace mock
}  // namespace mobilenet
