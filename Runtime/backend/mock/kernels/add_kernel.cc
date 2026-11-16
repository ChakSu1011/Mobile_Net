// 模块：Mock Add Kernel
//
// 职责：
//   实现逐元素加法的 CPU 参考计算：out = a + b。
//
// 边界：
//   - 只支持两个输入同形状、同 dtype；
//   - FP32 相加后按 half 舍入写回；
//   - 不做广播、原地写或并行。
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

// CPU 参考 Add Kernel。
class AddKernel final : public Kernel {
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

    const std::vector<int64_t> first = Dims(context.inputs[0].shape);
    const std::vector<int64_t> second = Dims(context.inputs[1].shape);
    const std::vector<int64_t> output = Dims(context.outputs[0].shape);
    if (first != second || first != output) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "AddKernel: tensor shape mismatch");
    }
    const int64_t total = NumElements(context.inputs[0].shape);
    for (int64_t i = 0; i < total; ++i) {
      WriteHalf(context.outputs[0], static_cast<uint64_t>(i),
                ReadHalf(context.inputs[0], static_cast<uint64_t>(i)) +
                    ReadHalf(context.inputs[1], static_cast<uint64_t>(i)));
    }
    return Status::Ok();
  }
};

}  // namespace

Status CreateAddKernel(std::unique_ptr<Kernel>* kernel) {
  if (kernel == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "CreateAddKernel: kernel output must not be null");
  }
  *kernel = std::make_unique<AddKernel>();
  return Status::Ok();
}

}  // namespace mock
}  // namespace mobilenet
