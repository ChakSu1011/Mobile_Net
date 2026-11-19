// 模块：RMSNorm 算子规格
//
// 职责：
//   定义最后一维 RMSNorm 的静态契约：half 输入与权重、half 输出；
//   校验 axis=-1、normalized_shape、eps 与形状一致性，组装 KernelDesc。
//
// 边界：
//   - 不支持 axis 非 -1、非 half 精度或 normalized_shape 不等于末维；
//   - 不做归一化数学计算。
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

// 校验 RMSNorm 的 attr、精度与形状。
Status ValidateRmsNorm(const runtime_graph::Op& op) {
  const auto& sources = op.SrcTensors();
  const auto& destinations = op.DstTensors();
  if (sources.size() != 2 || destinations.size() != 1) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "op=" + op.Name() +
                          ": rms_norm requires 2 inputs and 1 output");
  }
  I64 axis = -1;
  Status status = GetIntAttr(op, "axis", &axis);
  if (!status.ok()) {
    return status;
  }
  if (axis != -1) {
    return MakeStatus(
        ErrorCode::kUnsupportedOp,
        "op=" + op.Name() + ": rms_norm only supports axis=-1");
  }
  I64 normalized_shape = 0;
  status = GetIntAttr(op, "normalized_shape", &normalized_shape);
  if (!status.ok()) {
    return status;
  }
  F64 eps = 0.0;
  status = GetFloatAttr(op, "eps", &eps);
  if (!status.ok()) {
    return status;
  }
  if (normalized_shape <= 0 || eps < 0.0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "op=" + op.Name() + ": invalid rms_norm attrs");
  }

  const C8* op_name = op.Name().c_str();
  status = CheckTensor(*sources[0], runtime_graph::Precision::kFloat16,
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
  status = CheckRank(*sources[1], 1, op_name, "input[1]");
  if (!status.ok()) {
    return status;
  }

  const std::vector<I64> input =
      ShapeToDims(sources[0]->GetStorage().shape);
  const std::vector<I64> weight =
      ShapeToDims(sources[1]->GetStorage().shape);
  const std::vector<I64> output =
      ShapeToDims(destinations[0]->GetStorage().shape);
  if (input.empty() || weight[0] != normalized_shape ||
      input.back() != normalized_shape || output != input) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "op=" + op.Name() + ": rms_norm shape mismatch");
  }
  return Status::Ok();
}

// 组装 RMSNorm 的 KernelDesc。
Status BuildRmsNormDesc(const runtime_graph::Op& op, KernelDesc* desc) {
  return BuildKernelDesc(op, "rms_norm", *op.DstTensors()[0], desc);
}

}  // namespace

const OpSpec kRmsNormSpec = {
    runtime_graph::OpType::kRmsNorm,
    "rms_norm",
    false,
    0,
    &ValidateRmsNorm,
    &BuildRmsNormDesc,
};

}  // namespace mobilenet
