// 模块：Mock Gather Kernel
//
// 职责：
//   实现 token embedding 的 CPU 参考计算：out[b,s,:] = weight[ids[b,s],:]。
//
// 边界：
//   - 只支持 rank-2 int64 ids 与 rank-2 half 权重；
//   - 不处理越界静默修正；越界返回 kKernelError；
//   - 不做批量优化或并行。
//
// 依赖：
//   Runtime/backend/mock/kernels/kernel_common.h、mock_kernel_creators.h。

#include "Runtime/backend/mock/kernels/mock_kernel_creators.h"

#include <cstring>
#include <memory>
#include <vector>

#include "Runtime/backend/mock/kernels/kernel_common.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace mock {
namespace {

// CPU 参考 Gather Kernel。
class GatherKernel final : public Kernel {
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

    const std::vector<I64> ids = Dims(context.inputs[0].shape);
    const std::vector<I64> weight = Dims(context.inputs[1].shape);
    const std::vector<I64> output = Dims(context.outputs[0].shape);
    if (ids.size() != 2 || weight.size() != 2 || output.size() != 3) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "GatherKernel: unexpected tensor rank");
    }
    const I64 batch = ids[0];
    const I64 sequence = ids[1];
    const I64 vocab = weight[0];
    const I64 hidden = weight[1];
    if (batch < 0 || sequence < 0 || vocab <= 0 || hidden <= 0) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "GatherKernel: invalid shape");
    }
    if (output[0] != batch || output[1] != sequence ||
        output[2] != hidden) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "GatherKernel: output shape mismatch");
    }

    const I64* ids_data = Int64Data(context.inputs[0]);
    const U16* weight_data = HalfData(context.inputs[1]);
    U16* output_data = MutableHalfData(context.outputs[0]);
    if (ids_data == nullptr || weight_data == nullptr ||
        output_data == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "GatherKernel: null tensor data");
    }

    const I64 token_count = batch * sequence;
    const Usize row_bytes =
        static_cast<Usize>(hidden) * sizeof(U16);
    for (I64 token = 0; token < token_count; ++token) {
      const I64 id = ids_data[token];
      if (id < 0 || id >= vocab) {
        return MakeStatus(ErrorCode::kKernelError,
                          "GatherKernel: token id out of range");
      }
      std::memcpy(output_data + token * hidden,
                  weight_data + id * hidden, row_bytes);
    }
    return Status::Ok();
  }
};

}  // namespace

Status CreateGatherKernel(std::unique_ptr<Kernel>* kernel) {
  if (kernel == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "CreateGatherKernel: kernel output must not be null");
  }
  *kernel = std::make_unique<GatherKernel>();
  return Status::Ok();
}

}  // namespace mock
}  // namespace mobilenet
