// 模块：Mock RMSNorm Kernel
//
// 职责：
//   实现最后一维 RMSNorm 的 CPU 参考计算：
//   y = x * rsqrt(mean(x^2) + eps) * weight。
//
// 边界：
//   - 只支持 row_major、最后一维归一化；
//   - FP32 累加，FP16 写回；
//   - 不处理 axis 非 -1 或 normalized_shape 不等于末维的情况。
//
// 依赖：
//   Runtime/backend/mock/kernels/kernel_common.h、
//   Runtime/backend/ops/attr_utils.h、mock_kernel_creators.h。

#include "Runtime/backend/mock/kernels/mock_kernel_creators.h"

#include <cmath>
#include <memory>
#include <vector>

#include "Runtime/backend/mock/half_utils.h"
#include "Runtime/backend/mock/kernels/kernel_common.h"
#include "Runtime/backend/ops/attr_utils.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace mock {
namespace {

// CPU 参考 RMSNorm Kernel。
class RmsNormKernel final : public Kernel {
 public:
  Status Execute(const runtime_graph::KernelContext& context) override {
    Status status = RequireInputCount(context, 2);
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
    const std::vector<I64> output = Dims(context.outputs[0].shape);
    if (input.empty() || weight.size() != 1 || output != input) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "RmsNormKernel: unexpected tensor shape");
    }
    const I64 last_dim = input.back();
    if (last_dim <= 0 || weight[0] != last_dim) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "RmsNormKernel: normalized shape mismatch");
    }

    FP64 eps = 0.0;
    status = GetFloatAttr(*context.attrs, "eps", &eps);
    if (!status.ok()) {
      return status;
    }
    I64 normalized_shape = 0;
    status = GetIntAttr(*context.attrs, "normalized_shape",
                        &normalized_shape);
    if (!status.ok()) {
      return status;
    }
    if (normalized_shape != last_dim) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "RmsNormKernel: normalized_shape != last dim");
    }

    const U16* weight_data = HalfData(context.inputs[1]);
    if (weight_data == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "RmsNormKernel: null weight data");
    }
    const I64 rows = NumElements(context.inputs[0].shape) / last_dim;
    for (I64 row = 0; row < rows; ++row) {
      FP32 sum = 0.0f;
      for (I64 i = 0; i < last_dim; ++i) {
        const FP32 value = ReadHalf(context.inputs[0], row * last_dim + i);
        sum += value * value;
      }
      const FP32 mean = sum / static_cast<FP32>(last_dim);
      const FP32 scale = 1.0f / std::sqrt(mean + static_cast<FP32>(eps));
      for (I64 i = 0; i < last_dim; ++i) {
        const FP32 value = ReadHalf(context.inputs[0], row * last_dim + i);
        const FP32 weight_value = HalfToFloat(weight_data[i]);
        WriteHalf(context.outputs[0], row * last_dim + i,
                  value * scale * weight_value);
      }
    }
    return Status::Ok();
  }
};

}  // namespace

Status CreateRmsNormKernel(std::unique_ptr<Kernel>* kernel) {
  if (kernel == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "CreateRmsNormKernel: kernel output must not be null");
  }
  *kernel = std::make_unique<RmsNormKernel>();
  return Status::Ok();
}

}  // namespace mock
}  // namespace mobilenet
