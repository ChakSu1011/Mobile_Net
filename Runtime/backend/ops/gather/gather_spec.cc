// 模块：Gather 算子规格
//
// 职责：
//   定义 token embedding 的静态契约：两个输入（int64 ids、half 权重）、
//   一个 half 输出；校验 axis=0、秩与形状一致性，并组装 KernelDesc。
//
// 边界：
//   - 不做 gather 数学计算；
//   - 不支持 axis 非 0、ids 越界静默修正或非 half 权重。
//
// 依赖：
//   Runtime/backend/ops/op_specs.h、attr_utils.h、op_validation.h。

#include <string>
#include <vector>

#include "Graph/include/tensor.h"
#include "Runtime/backend/ops/attr_utils.h"
#include "Runtime/backend/ops/op_specs.h"
#include "Runtime/backend/ops/op_validation.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace {

// 校验 Gather 的 attr、精度与形状。
Status ValidateGather(const runtime_graph::Op& op) {
  const auto& sources = op.SrcTensors();
  const auto& destinations = op.DstTensors();
  if (sources.size() != 2 || destinations.size() != 1) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "op=" + op.Name() +
                          ": gather requires 2 inputs and 1 output");
  }
  I64 axis = 0;
  Status status = GetIntAttr(op, "axis", &axis);
  if (!status.ok()) {
    return status;
  }
  if (axis != 0) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "op=" + op.Name() + ": gather only supports axis=0");
  }
  const C8* op_name = op.Name().c_str();
  status = CheckTensor(*sources[0], runtime_graph::Precision::kInt64,
                       op_name, "input[0]");
  if (!status.ok()) {
    return status;
  }
  status = CheckTensor(*sources[1], runtime_graph::Precision::kFloat16,
                       op_name, "input[1]");
  if (!status.ok()) {
    return status;
  }
  status = CheckTensor(*destinations[0],
                       runtime_graph::Precision::kFloat16, op_name,
                       "output[0]");
  if (!status.ok()) {
    return status;
  }
  status = CheckRank(*sources[0], 2, op_name, "input[0]");
  if (!status.ok()) {
    return status;
  }
  status = CheckRank(*sources[1], 2, op_name, "input[1]");
  if (!status.ok()) {
    return status;
  }
  status = CheckRank(*destinations[0], 3, op_name, "output[0]");
  if (!status.ok()) {
    return status;
  }

  const std::vector<I64> ids =
      ShapeToDims(sources[0]->GetStorage().shape);
  const std::vector<I64> weight =
      ShapeToDims(sources[1]->GetStorage().shape);
  const std::vector<I64> output =
      ShapeToDims(destinations[0]->GetStorage().shape);
  if (output[0] != ids[0] || output[1] != ids[1] ||
      output[2] != weight[1]) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "op=" + op.Name() + ": gather output shape mismatch");
  }
  return Status::Ok();
}

// 组装 Gather 的 KernelDesc。
Status BuildGatherDesc(const runtime_graph::Op& op, KernelDesc* desc) {
  return BuildKernelDesc(op, "gather", *op.DstTensors()[0], desc);
}

}  // namespace

const OpSpec kGatherSpec = {
    runtime_graph::OpType::kGather,
    "gather",
    false,
    0,
    &ValidateGather,
    &BuildGatherDesc,
};

}  // namespace mobilenet
