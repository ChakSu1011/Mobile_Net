// 模块：Mock Sample Kernel
//
// 职责：
//   实现 argmax 采样：对 [B,S,V] logits 的最后一维取最大值下标，输出
//   [B,S] int64 token id。
//
// 边界：
//   - 只支持 method=argmax、axis=-1；
//   - 并列最大值取第一个下标；
//   - 不做 top-k/top-p/temperature 等采样算法。
//
// 依赖：
//   Runtime/backend/mock/kernels/kernel_common.h、mock_kernel_creators.h。

#include "Runtime/backend/mock/kernels/mock_kernel_creators.h"

#include <memory>
#include <vector>

#include "Runtime/backend/mock/half_utils.h"
#include "Runtime/backend/mock/kernels/kernel_common.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace mock {
namespace {

// CPU 参考 Argmax Sample Kernel。
class SampleKernel final : public Kernel {
 public:
  Status Execute(const runtime_graph::KernelContext& context) override {
    Status status = RequireInputCount(context, 1);
    if (!status.ok()) {
      return status;
    }
    status = RequireOutputCount(context, 1);
    if (!status.ok()) {
      return status;
    }

    const std::vector<I64> logits = Dims(context.inputs[0].shape);
    const std::vector<I64> output = Dims(context.outputs[0].shape);
    if (logits.size() < 2 || output.size() + 1 != logits.size()) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "SampleKernel: unexpected tensor rank");
    }
    for (Usize i = 0; i < output.size(); ++i) {
      if (output[i] != logits[i]) {
        return MakeStatus(ErrorCode::kInvalidArgument,
                          "SampleKernel: output shape mismatch");
      }
    }
    const I64 vocab = logits.back();
    if (vocab <= 0) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "SampleKernel: empty vocabulary");
    }

    const I64 rows = NumElements(context.inputs[0].shape) / vocab;
    I64* output_data = MutableInt64Data(context.outputs[0]);
    if (output_data == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "SampleKernel: null output data");
    }
    for (I64 row = 0; row < rows; ++row) {
      const I64 base = row * vocab;
      FP32 best = ReadHalf(context.inputs[0], static_cast<U64>(base));
      I64 best_index = 0;
      for (I64 i = 1; i < vocab; ++i) {
        const FP32 value =
            ReadHalf(context.inputs[0], static_cast<U64>(base + i));
        if (value > best) {
          best = value;
          best_index = i;
        }
      }
      output_data[row] = best_index;
    }
    return Status::Ok();
  }
};

}  // namespace

Status CreateSampleKernel(std::unique_ptr<Kernel>* kernel) {
  if (kernel == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "CreateSampleKernel: kernel output must not be null");
  }
  *kernel = std::make_unique<SampleKernel>();
  return Status::Ok();
}

}  // namespace mock
}  // namespace mobilenet
