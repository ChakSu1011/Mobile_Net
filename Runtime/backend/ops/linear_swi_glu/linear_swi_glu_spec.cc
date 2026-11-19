// 模块：LinearSwiGlu 算子规格
//
// 职责：
//   定义 gate/up 双线性 + SiLU 门控的静态契约：half 输入、gate/up 权重、
//   half 输出；校验 intermediate_size/activation/transpose_weight 与形状
//   一致性，组装 KernelDesc。
//
// 边界：
//   - 只支持 activation=silu、transpose_weight=true、bias=false；
//   - 不支持非 half 精度或 gate/up 形状不一致。
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

// 校验 LinearSwiGlu 的 attr、精度与形状。
Status ValidateLinearSwiGlu(const runtime_graph::Op& op) {
  const auto& sources = op.SrcTensors();
  const auto& destinations = op.DstTensors();
  if (sources.size() != 3 || destinations.size() != 1) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "op=" + op.Name() +
            ": linear_swi_glu requires 3 inputs and 1 output");
  }
  I64 intermediate_size = 0;
  Status status = GetIntAttr(op, "intermediate_size", &intermediate_size);
  if (!status.ok()) {
    return status;
  }
  std::string activation;
  status = GetStringAttr(op, "activation", &activation);
  if (!status.ok()) {
    return status;
  }
  if (activation != "silu") {
    return MakeStatus(
        ErrorCode::kUnsupportedOp,
        "op=" + op.Name() + ": linear_swi_glu only supports silu");
  }
  Bool transpose = true;
  status = GetBoolAttr(op, "transpose_weight", &transpose);
  if (!status.ok()) {
    return status;
  }
  if (!transpose) {
    return MakeStatus(
        ErrorCode::kUnsupportedOp,
        "op=" + op.Name() +
            ": linear_swi_glu only supports transpose_weight=true");
  }
  if (HasAttr(op.Attrs(), "bias")) {
    Bool bias = false;
    status = GetBoolAttr(op, "bias", &bias);
    if (!status.ok()) {
      return status;
    }
    if (bias) {
      return MakeStatus(
          ErrorCode::kUnsupportedOp,
          "op=" + op.Name() + ": linear_swi_glu bias is unsupported");
    }
  }
  if (intermediate_size <= 0) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "op=" + op.Name() + ": invalid intermediate_size");
  }

  const C8* op_name = op.Name().c_str();
  for (Usize i = 0; i < sources.size(); ++i) {
    status = CheckTensor(*sources[i],
                         runtime_graph::Precision::kFloat16, op_name,
                         "input");
    if (!status.ok()) {
      return status;
    }
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
  status = CheckRank(*sources[2], 2, op_name, "input[2]");
  if (!status.ok()) {
    return status;
  }

  const std::vector<I64> input =
      ShapeToDims(sources[0]->GetStorage().shape);
  const std::vector<I64> gate =
      ShapeToDims(sources[1]->GetStorage().shape);
  const std::vector<I64> up =
      ShapeToDims(sources[2]->GetStorage().shape);
  const std::vector<I64> output =
      ShapeToDims(destinations[0]->GetStorage().shape);
  if (input.empty() || gate != up || output.size() != input.size()) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "op=" + op.Name() + ": linear_swi_glu rank mismatch");
  }
  if (gate[0] != intermediate_size || gate[1] != input.back() ||
      output.back() != intermediate_size) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "op=" + op.Name() + ": linear_swi_glu shape mismatch");
  }
  status = CheckSamePrefix(input, output, input.size() - 1,
                           "linear_swi_glu output prefix");
  if (!status.ok()) {
    return status;
  }
  return Status::Ok();
}

// 组装 LinearSwiGlu 的 KernelDesc。
Status BuildLinearSwiGluDesc(const runtime_graph::Op& op,
                             KernelDesc* desc) {
  return BuildKernelDesc(op, "linear_swi_glu", *op.DstTensors()[0],
                         desc);
}

}  // namespace

const OpSpec kLinearSwiGluSpec = {
    runtime_graph::OpType::kLinearSwiGlu,
    "linear_swi_glu",
    false,
    0,
    &ValidateLinearSwiGlu,
    &BuildLinearSwiGluDesc,
};

}  // namespace mobilenet
