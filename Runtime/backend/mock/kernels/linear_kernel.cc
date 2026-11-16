// 模块：Mock Linear Kernel
//
// 职责：
//   实现 y = x · W^T 的 CPU 参考计算，W 布局为 [out_features, in_features]。
//
// 边界：
//   - 只支持 transpose_weight=true、无 bias；
//   - FP32 累加，FP16 写回；
//   - 不做分块、向量化或并行。
//
// 依赖：
//   Runtime/backend/mock/kernels/kernel_common.h、mock_kernel_creators.h。

#include "Runtime/backend/mock/kernels/mock_kernel_creators.h"

#include <memory>
#include <vector>

#include "Runtime/backend/mock/kernels/kernel_common.h"

namespace mobilenet {
namespace mock {
namespace {

// CPU 参考 Linear Kernel。
class LinearKernel final : public Kernel {
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

    const std::vector<int64_t> input = Dims(context.inputs[0].shape);
    const std::vector<int64_t> weight = Dims(context.inputs[1].shape);
    const std::vector<int64_t> output = Dims(context.outputs[0].shape);
    if (input.empty() || weight.size() != 2 ||
        output.size() != input.size()) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "LinearKernel: unexpected tensor rank");
    }
    const int64_t in_features = input.back();
    const int64_t out_features = weight[0];
    if (in_features <= 0 || out_features <= 0 ||
        weight[1] != in_features) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "LinearKernel: weight shape mismatch");
    }
    if (output.back() != out_features) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "LinearKernel: output last dim mismatch");
    }
    for (size_t i = 0; i + 1 < input.size(); ++i) {
      if (output[i] != input[i]) {
        return MakeStatus(ErrorCode::kInvalidArgument,
                          "LinearKernel: output prefix mismatch");
      }
    }

    const int64_t rows = NumElements(context.inputs[0].shape) / in_features;
    for (int64_t row = 0; row < rows; ++row) {
      const int64_t input_base = row * in_features;
      for (int64_t out = 0; out < out_features; ++out) {
        const int64_t weight_base = out * in_features;
        float sum = 0.0f;
        for (int64_t i = 0; i < in_features; ++i) {
          sum += ReadHalf(context.inputs[0], input_base + i) *
                 ReadHalf(context.inputs[1], weight_base + i);
        }
        WriteHalf(context.outputs[0], row * out_features + out, sum);
      }
    }
    return Status::Ok();
  }
};

}  // namespace

Status CreateLinearKernel(std::unique_ptr<Kernel>* kernel) {
  if (kernel == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "CreateLinearKernel: kernel output must not be null");
  }
  *kernel = std::make_unique<LinearKernel>();
  return Status::Ok();
}

}  // namespace mock
}  // namespace mobilenet
