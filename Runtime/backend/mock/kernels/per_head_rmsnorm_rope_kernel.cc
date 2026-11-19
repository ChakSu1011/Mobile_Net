// 模块：Mock LinearPerHeadRMSNormRope Kernel
//
// 职责：
//   实现 linear + per-head RMSNorm + NeoX RoPE 融合算子的 CPU 参考计算：
//   y = rope(per_head_rms_norm(x · W^T))。
//
// 边界：
//   - 只支持 rope_layout=neox、transpose_weight=true、无 bias；
//   - positions 为 rank-(input_rank-1) 的 int64 张量；
//   - FP32 计算，FP16 写回；
//   - 不做分块、向量化或并行。
//
// 依赖：
//   Runtime/backend/mock/kernels/kernel_common.h、
//   Runtime/backend/ops/attr_utils.h、mock_kernel_creators.h。

#include "Runtime/backend/mock/kernels/mock_kernel_creators.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "Runtime/backend/mock/half_utils.h"
#include "Runtime/backend/mock/kernels/kernel_common.h"
#include "Runtime/backend/ops/attr_utils.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace mock {
namespace {

// CPU 参考 Q/K path Kernel。
class PerHeadRmsNormRopeKernel final : public Kernel {
 public:
  Status Execute(const runtime_graph::KernelContext& context) override {
    Status status = RequireInputCount(context, 4);
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

    const std::vector<I64> input = Dims(context.inputs[0].shape);
    const std::vector<I64> weight = Dims(context.inputs[1].shape);
    const std::vector<I64> norm = Dims(context.inputs[2].shape);
    const std::vector<I64> positions = Dims(context.inputs[3].shape);
    const std::vector<I64> output = Dims(context.outputs[0].shape);
    if (input.empty() || weight.size() != 2 || norm.size() != 1 ||
        positions.size() + 1 != input.size() ||
        output.size() != input.size()) {
      return MakeStatus(
          ErrorCode::kInvalidArgument,
          "PerHeadRmsNormRopeKernel: unexpected tensor rank");
    }
    const I64 in_features = input.back();
    const I64 out_features = weight[0];
    if (in_features <= 0 || out_features <= 0 ||
        weight[1] != in_features || output.back() != out_features) {
      return MakeStatus(
          ErrorCode::kInvalidArgument,
          "PerHeadRmsNormRopeKernel: weight/output shape mismatch");
    }
    for (Usize i = 0; i + 1 < input.size(); ++i) {
      if (output[i] != input[i]) {
        return MakeStatus(
            ErrorCode::kInvalidArgument,
            "PerHeadRmsNormRopeKernel: output prefix mismatch");
      }
    }

    I64 num_heads = 0;
    I64 head_dim = 0;
    FP64 norm_eps = 0.0;
    FP64 rope_theta = 0.0;
    std::string rope_layout;
    status = GetIntAttr(*context.attrs, "num_heads", &num_heads);
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
    status = GetStringAttr(*context.attrs, "rope_layout", &rope_layout);
    if (!status.ok()) {
      return status;
    }
    if (rope_layout != "neox") {
      return MakeStatus(ErrorCode::kUnsupportedOp,
                        "PerHeadRmsNormRopeKernel: only neox layout");
    }
    if (num_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0) {
      return MakeStatus(
          ErrorCode::kInvalidArgument,
          "PerHeadRmsNormRopeKernel: invalid head configuration");
    }
    if (out_features != num_heads * head_dim || norm[0] != head_dim) {
      return MakeStatus(
          ErrorCode::kInvalidArgument,
          "PerHeadRmsNormRopeKernel: head/norm shape mismatch");
    }

    const I64 rows = NumElements(context.inputs[0].shape) / in_features;
    if (NumElements(context.inputs[3].shape) != rows) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "PerHeadRmsNormRopeKernel: positions size mismatch");
    }
    const I64* position_data = Int64Data(context.inputs[3]);
    const U16* norm_data = HalfData(context.inputs[2]);
    if (position_data == nullptr || norm_data == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "PerHeadRmsNormRopeKernel: null tensor data");
    }

    std::vector<FP32> values(static_cast<Usize>(out_features), 0.0f);
    const I64 half_dim = head_dim / 2;
    for (I64 row = 0; row < rows; ++row) {
      const I64 input_base = row * in_features;
      for (I64 out = 0; out < out_features; ++out) {
        const I64 weight_base = out * in_features;
        FP32 sum = 0.0f;
        for (I64 i = 0; i < in_features; ++i) {
          sum += ReadHalf(context.inputs[0], input_base + i) *
                 ReadHalf(context.inputs[1], weight_base + i);
        }
        values[static_cast<Usize>(out)] = sum;
      }

      for (I64 head = 0; head < num_heads; ++head) {
        const I64 base = head * head_dim;
        FP32 square_sum = 0.0f;
        for (I64 i = 0; i < head_dim; ++i) {
          const FP32 value = values[static_cast<Usize>(base + i)];
          square_sum += value * value;
        }
        const FP32 mean = square_sum / static_cast<FP32>(head_dim);
        const FP32 scale =
            1.0f / std::sqrt(mean + static_cast<FP32>(norm_eps));
        for (I64 i = 0; i < head_dim; ++i) {
          values[static_cast<Usize>(base + i)] *=
              scale * HalfToFloat(norm_data[i]);
        }
      }

      const I64 position = position_data[row];
      for (I64 head = 0; head < num_heads; ++head) {
        const I64 base = head * head_dim;
        for (I64 i = 0; i < half_dim; ++i) {
          const FP64 exponent =
              -2.0 * static_cast<FP64>(i) /
              static_cast<FP64>(head_dim);
          const FP64 angle =
              static_cast<FP64>(position) *
              std::pow(rope_theta, exponent);
          const FP32 cosine = static_cast<FP32>(std::cos(angle));
          const FP32 sine = static_cast<FP32>(std::sin(angle));
          const FP32 first = values[static_cast<Usize>(base + i)];
          const FP32 second =
              values[static_cast<Usize>(base + i + half_dim)];
          values[static_cast<Usize>(base + i)] =
              first * cosine - second * sine;
          values[static_cast<Usize>(base + i + half_dim)] =
              second * cosine + first * sine;
        }
      }

      for (I64 out = 0; out < out_features; ++out) {
        WriteHalf(context.outputs[0], row * out_features + out,
                  values[static_cast<Usize>(out)]);
      }
    }
    return Status::Ok();
  }
};

}  // namespace

Status CreatePerHeadRmsNormRopeKernel(std::unique_ptr<Kernel>* kernel) {
  if (kernel == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "CreatePerHeadRmsNormRopeKernel: kernel output must not be null");
  }
  *kernel = std::make_unique<PerHeadRmsNormRopeKernel>();
  return Status::Ok();
}

}  // namespace mock
}  // namespace mobilenet
