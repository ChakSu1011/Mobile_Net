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

    const std::vector<int64_t> input = Dims(context.inputs[0].shape);
    const std::vector<int64_t> weight = Dims(context.inputs[1].shape);
    const std::vector<int64_t> norm = Dims(context.inputs[2].shape);
    const std::vector<int64_t> positions = Dims(context.inputs[3].shape);
    const std::vector<int64_t> output = Dims(context.outputs[0].shape);
    if (input.empty() || weight.size() != 2 || norm.size() != 1 ||
        positions.size() + 1 != input.size() ||
        output.size() != input.size()) {
      return MakeStatus(
          ErrorCode::kInvalidArgument,
          "PerHeadRmsNormRopeKernel: unexpected tensor rank");
    }
    const int64_t in_features = input.back();
    const int64_t out_features = weight[0];
    if (in_features <= 0 || out_features <= 0 ||
        weight[1] != in_features || output.back() != out_features) {
      return MakeStatus(
          ErrorCode::kInvalidArgument,
          "PerHeadRmsNormRopeKernel: weight/output shape mismatch");
    }
    for (size_t i = 0; i + 1 < input.size(); ++i) {
      if (output[i] != input[i]) {
        return MakeStatus(
            ErrorCode::kInvalidArgument,
            "PerHeadRmsNormRopeKernel: output prefix mismatch");
      }
    }

    int64_t num_heads = 0;
    int64_t head_dim = 0;
    double norm_eps = 0.0;
    double rope_theta = 0.0;
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

    const int64_t rows = NumElements(context.inputs[0].shape) / in_features;
    if (NumElements(context.inputs[3].shape) != rows) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "PerHeadRmsNormRopeKernel: positions size mismatch");
    }
    const int64_t* position_data = Int64Data(context.inputs[3]);
    const uint16_t* norm_data = HalfData(context.inputs[2]);
    if (position_data == nullptr || norm_data == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "PerHeadRmsNormRopeKernel: null tensor data");
    }

    std::vector<float> values(static_cast<size_t>(out_features), 0.0f);
    const int64_t half_dim = head_dim / 2;
    for (int64_t row = 0; row < rows; ++row) {
      const int64_t input_base = row * in_features;
      for (int64_t out = 0; out < out_features; ++out) {
        const int64_t weight_base = out * in_features;
        float sum = 0.0f;
        for (int64_t i = 0; i < in_features; ++i) {
          sum += ReadHalf(context.inputs[0], input_base + i) *
                 ReadHalf(context.inputs[1], weight_base + i);
        }
        values[static_cast<size_t>(out)] = sum;
      }

      for (int64_t head = 0; head < num_heads; ++head) {
        const int64_t base = head * head_dim;
        float square_sum = 0.0f;
        for (int64_t i = 0; i < head_dim; ++i) {
          const float value = values[static_cast<size_t>(base + i)];
          square_sum += value * value;
        }
        const float mean = square_sum / static_cast<float>(head_dim);
        const float scale =
            1.0f / std::sqrt(mean + static_cast<float>(norm_eps));
        for (int64_t i = 0; i < head_dim; ++i) {
          values[static_cast<size_t>(base + i)] *=
              scale * HalfToFloat(norm_data[i]);
        }
      }

      const int64_t position = position_data[row];
      for (int64_t head = 0; head < num_heads; ++head) {
        const int64_t base = head * head_dim;
        for (int64_t i = 0; i < half_dim; ++i) {
          const double exponent =
              -2.0 * static_cast<double>(i) /
              static_cast<double>(head_dim);
          const double angle =
              static_cast<double>(position) *
              std::pow(rope_theta, exponent);
          const float cosine = static_cast<float>(std::cos(angle));
          const float sine = static_cast<float>(std::sin(angle));
          const float first = values[static_cast<size_t>(base + i)];
          const float second =
              values[static_cast<size_t>(base + i + half_dim)];
          values[static_cast<size_t>(base + i)] =
              first * cosine - second * sine;
          values[static_cast<size_t>(base + i + half_dim)] =
              second * cosine + first * sine;
        }
      }

      for (int64_t out = 0; out < out_features; ++out) {
        WriteHalf(context.outputs[0], row * out_features + out,
                  values[static_cast<size_t>(out)]);
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
