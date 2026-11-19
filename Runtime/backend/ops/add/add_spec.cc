// 模块：Add 算子规格
//
// 职责：
//   定义逐元素加法的静态契约：两个 half 输入、一个 half 输出；校验形状
//   完全一致，组装 KernelDesc。
//
// 边界：
//   - 不支持广播、非 half 精度或不同形状；
//   - 不做加法数学计算。
//
// 依赖：
//   Runtime/backend/ops/op_specs.h、op_validation.h。

#include <string>
#include <vector>

#include "Graph/include/tensor.h"
#include "Runtime/backend/ops/op_specs.h"
#include "Runtime/backend/ops/op_validation.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace {

// 校验 Add 的精度与形状。
Status ValidateAdd(const runtime_graph::Op& op) {
  const auto& sources = op.SrcTensors();
  const auto& destinations = op.DstTensors();
  if (sources.size() != 2 || destinations.size() != 1) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "op=" + op.Name() +
                          ": add requires 2 inputs and 1 output");
  }
  const C8* op_name = op.Name().c_str();
  for (Usize i = 0; i < sources.size(); ++i) {
    const Status status = CheckTensor(
        *sources[i], runtime_graph::Precision::kFloat16, op_name, "input");
    if (!status.ok()) {
      return status;
    }
  }
  Status status = CheckTensor(*destinations[0],
                              runtime_graph::Precision::kFloat16, op_name,
                              "output[0]");
  if (!status.ok()) {
    return status;
  }
  const std::vector<I64> first =
      ShapeToDims(sources[0]->GetStorage().shape);
  const std::vector<I64> second =
      ShapeToDims(sources[1]->GetStorage().shape);
  const std::vector<I64> output =
      ShapeToDims(destinations[0]->GetStorage().shape);
  if (first != second || first != output) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "op=" + op.Name() + ": add shape mismatch");
  }
  return Status::Ok();
}

// 组装 Add 的 KernelDesc。
Status BuildAddDesc(const runtime_graph::Op& op, KernelDesc* desc) {
  return BuildKernelDesc(op, "add", *op.DstTensors()[0], desc);
}

}  // namespace

const OpSpec kAddSpec = {
    runtime_graph::OpType::kAdd,
    "add",
    false,
    0,
    &ValidateAdd,
    &BuildAddDesc,
};

}  // namespace mobilenet
