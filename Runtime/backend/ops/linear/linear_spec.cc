// 模块：Linear 算子规格
//
// 职责：
//   定义 y = x · W^T 的静态契约：half 输入/权重/输出；校验
//   in_features/out_features、transpose_weight、无 bias 与形状一致性，
//   组装 KernelDesc。
//
// 边界：
//   - 只支持 transpose_weight=true、bias=false；
//   - 不支持非 half 精度、广播或 rank<1 输入。
//
// 依赖：
//   Runtime/backend/ops/op_specs.h、attr_utils.h、op_validation.h。

#include <string>
#include <vector>

#include "Graph/include/tensor.h"
#include "Runtime/backend/ops/attr_utils.h"
#include "Runtime/backend/ops/op_specs.h"
#include "Runtime/backend/ops/op_validation.h"

namespace mobilenet {
namespace {

// 校验 Linear 的 attr、精度与形状。
Status ValidateLinear(const runtime_graph::Op& op) {
  const auto& sources = op.SrcTensors();
  const auto& destinations = op.DstTensors();
  if (sources.size() != 2 || destinations.size() != 1) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "op=" + op.Name() +
                          ": linear requires 2 inputs and 1 output");
  }
  int64_t in_features = 0;
  int64_t out_features = 0;
  Status status = GetIntAttr(op, "in_features", &in_features);
  if (!status.ok()) {
    return status;
  }
  status = GetIntAttr(op, "out_features", &out_features);
  if (!status.ok()) {
    return status;
  }
  bool transpose = true;
  status = GetBoolAttr(op, "transpose_weight", &transpose);
  if (!status.ok()) {
    return status;
  }
  if (!transpose) {
    return MakeStatus(
        ErrorCode::kUnsupportedOp,
        "op=" + op.Name() + ": linear only supports transpose_weight=true");
  }
  if (HasAttr(op.Attrs(), "bias")) {
    bool bias = false;
    status = GetBoolAttr(op, "bias", &bias);
    if (!status.ok()) {
      return status;
    }
    if (bias) {
      return MakeStatus(ErrorCode::kUnsupportedOp,
                        "op=" + op.Name() + ": linear bias is unsupported");
    }
  }
  if (in_features <= 0 || out_features <= 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "op=" + op.Name() + ": invalid linear features");
  }

  const char* op_name = op.Name().c_str();
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
  status = CheckRank(*sources[1], 2, op_name, "input[1]");
  if (!status.ok()) {
    return status;
  }

  const std::vector<int64_t> input =
      ShapeToDims(sources[0]->GetStorage().shape);
  const std::vector<int64_t> weight =
      ShapeToDims(sources[1]->GetStorage().shape);
  const std::vector<int64_t> output =
      ShapeToDims(destinations[0]->GetStorage().shape);
  if (input.empty() || output.size() != input.size()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "op=" + op.Name() + ": linear rank mismatch");
  }
  if (weight[0] != out_features || weight[1] != in_features ||
      input.back() != in_features || output.back() != out_features) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "op=" + op.Name() + ": linear feature shape mismatch");
  }
  status = CheckSamePrefix(input, output, input.size() - 1,
                           "linear output prefix");
  if (!status.ok()) {
    return status;
  }
  return Status::Ok();
}

// 组装 Linear 的 KernelDesc。
Status BuildLinearDesc(const runtime_graph::Op& op, KernelDesc* desc) {
  return BuildKernelDesc(op, "linear", *op.DstTensors()[0], desc);
}

}  // namespace

const OpSpec kLinearSpec = {
    runtime_graph::OpType::kLinear,
    "linear",
    false,
    0,
    &ValidateLinear,
    &BuildLinearDesc,
};

}  // namespace mobilenet
