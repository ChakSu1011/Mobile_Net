// 模块：Mock LinearSwiGlu Kernel
//
// 职责：
//   实现 gate/up 双线性 + SiLU 门控的 CPU 参考计算：
//   y = silu(x · gate^T) * (x · up^T)。
//
// 边界：
//   - 只支持 activation=silu、transpose_weight=true、无 bias；
//   - FP32 累加，FP16 写回；
//   - 不做分块、向量化或并行。
//
// 依赖：
//   Runtime/backend/mock/kernels/kernel_common.h、mock_kernel_creators.h。

#include "Runtime/backend/mock/kernels/mock_kernel_creators.h"

#include <cmath>
#include <memory>
#include <vector>

#include "Runtime/backend/mock/kernels/kernel_common.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace mock {
namespace {

// CPU 参考 LinearSwiGlu Kernel。
class LinearSwiGluKernel final : public Kernel {
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

    const std::vector<I64> input = Dims(context.inputs[0].shape);
    const std::vector<I64> gate = Dims(context.inputs[1].shape);
    const std::vector<I64> up = Dims(context.inputs[2].shape);
    const std::vector<I64> output = Dims(context.outputs[0].shape);
    if (input.empty() || gate.size() != 2 || up != gate ||
        output.size() != input.size()) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "LinearSwiGluKernel: unexpected tensor rank");
    }
    const I64 in_features = input.back();
    const I64 intermediate = gate[0];
    if (in_features <= 0 || intermediate <= 0 ||
        gate[1] != in_features) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "LinearSwiGluKernel: weight shape mismatch");
    }
    if (output.back() != intermediate) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "LinearSwiGluKernel: output last dim mismatch");
    }
    for (Usize i = 0; i + 1 < input.size(); ++i) {
      if (output[i] != input[i]) {
        return MakeStatus(
            ErrorCode::kInvalidArgument,
            "LinearSwiGluKernel: output prefix mismatch");
      }
    }

    const I64 rows = NumElements(context.inputs[0].shape) / in_features;
    for (I64 row = 0; row < rows; ++row) {
      const I64 input_base = row * in_features;
      for (I64 out = 0; out < intermediate; ++out) {
        const I64 weight_base = out * in_features;
        FP32 gate_sum = 0.0f;
        FP32 up_sum = 0.0f;
        for (I64 i = 0; i < in_features; ++i) {
          const FP32 value = ReadHalf(context.inputs[0], input_base + i);
          gate_sum += value * ReadHalf(context.inputs[1], weight_base + i);
          up_sum += value * ReadHalf(context.inputs[2], weight_base + i);
        }
        const FP32 activated = gate_sum / (1.0f + std::exp(-gate_sum));
        WriteHalf(context.outputs[0], row * intermediate + out,
                  activated * up_sum);
      }
    }
    return Status::Ok();
  }
};

}  // namespace

Status CreateLinearSwiGluKernel(std::unique_ptr<Kernel>* kernel) {
  if (kernel == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "CreateLinearSwiGluKernel: kernel output must not be null");
  }
  *kernel = std::make_unique<LinearSwiGluKernel>();
  return Status::Ok();
}

}  // namespace mock
}  // namespace mobilenet
