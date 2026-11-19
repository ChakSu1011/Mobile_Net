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

}  // namespace mock
}  // namespace mobilenet
