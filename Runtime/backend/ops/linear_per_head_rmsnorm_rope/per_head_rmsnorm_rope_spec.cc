// 模块：LinearPerHeadRMSNormRope 算子规格
//
// 职责：
//   定义 linear + per-head RMSNorm + NeoX RoPE 融合算子的静态契约：
//   half 输入/权重/norm、int64 positions、half 输出；校验 head 配置、
//   rope 配置与形状一致性，组装 KernelDesc。
//
// 边界：
//   - 只支持 rope_layout=neox、rope_type=default、transpose_weight=true、
//     bias=false；
//   - 不支持 scaling、非 half 精度或 head_dim 偶数以外的配置。
//
// 依赖：
//   Runtime/backend/ops/op_specs.h、attr_utils.h、op_validation.h。

#include <cmath>
#include <string>
#include <vector>

#include "Graph/include/tensor.h"
#include "Runtime/backend/ops/attr_utils.h"
#include "Runtime/backend/ops/op_specs.h"
#include "Runtime/backend/ops/op_validation.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace {

// 校验融合 Q/K path 的 attr、精度与形状。
Status ValidatePerHeadRmsNormRope(const runtime_graph::Op& op) {
  const auto& sources = op.SrcTensors();
  const auto& destinations = op.DstTensors();
  if (sources.size() != 4 || destinations.size() != 1) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "op=" + op.Name() +
            ": per_head_rmsnorm_rope requires 4 inputs and 1 output");
  }

  I64 in_features = 0;
  I64 out_features = 0;
  I64 num_heads = 0;
  I64 head_dim = 0;
  FP64 norm_eps = 0.0;
  FP64 rope_theta = 0.0;
  std::string rope_layout;
  Status status = GetIntAttr(op, "in_features", &in_features);
  if (!status.ok()) {
    return status;
  }
  status = GetIntAttr(op, "out_features", &out_features);
  if (!status.ok()) {
    return status;
  }
  status = GetIntAttr(op, "num_heads", &num_heads);
  if (!status.ok()) {
    return status;
  }
  status = GetIntAttr(op, "head_dim", &head_dim);
  if (!status.ok()) {
    return status;
  }
  status = GetFloatAttr(op, "norm_eps", &norm_eps);
  if (!status.ok()) {
    return status;
  }
  status = GetFloatAttr(op, "rope_theta", &rope_theta);
  if (!status.ok()) {
    return status;
  }
  status = GetStringAttr(op, "rope_layout", &rope_layout);
  if (!status.ok()) {
    return status;
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
            ": per_head_rmsnorm_rope only supports transpose_weight=true");
  }
  if (rope_layout != "neox") {
    return MakeStatus(
        ErrorCode::kUnsupportedOp,
        "op=" + op.Name() +
            ": per_head_rmsnorm_rope only supports neox");
  }
  if (HasAttr(op.Attrs(), "rope_type")) {
    std::string rope_type;
    status = GetStringAttr(op, "rope_type", &rope_type);
    if (!status.ok()) {
      return status;
    }
    if (rope_type != "default") {
      return MakeStatus(
          ErrorCode::kUnsupportedOp,
          "op=" + op.Name() +
              ": per_head_rmsnorm_rope only supports rope_type=default");
    }
  }
  if (HasAttr(op.Attrs(), "scaling_factor")) {
    FP64 scaling = 1.0;
    status = GetFloatAttr(op, "scaling_factor", &scaling);
    if (!status.ok()) {
      return status;
    }
    if (std::fabs(scaling - 1.0) > 1e-6) {
      return MakeStatus(
          ErrorCode::kUnsupportedOp,
          "op=" + op.Name() +
              ": per_head_rmsnorm_rope scaling is unsupported");
    }
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
          "op=" + op.Name() +
              ": per_head_rmsnorm_rope bias is unsupported");
    }
  }
  if (in_features <= 0 || out_features <= 0 || num_heads <= 0 ||
      head_dim <= 0 || head_dim % 2 != 0 || num_heads * head_dim !=
          out_features || norm_eps < 0.0 || rope_theta <= 0.0) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "op=" + op.Name() +
            ": invalid per_head_rmsnorm_rope head/rope attrs");
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
  status = CheckTensor(*sources[2], runtime_graph::Precision::kFloat16,
                       op_name, "input[2]");
  if (!status.ok()) {
    return status;
  }
  status = CheckTensor(*sources[3], runtime_graph::Precision::kInt64,
                       op_name, "input[3]");
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
  status = CheckRank(*sources[2], 1, op_name, "input[2]");
  if (!status.ok()) {
    return status;
  }

  const std::vector<I64> input =
      ShapeToDims(sources[0]->GetStorage().shape);
  const std::vector<I64> weight =
      ShapeToDims(sources[1]->GetStorage().shape);
  const std::vector<I64> norm =
      ShapeToDims(sources[2]->GetStorage().shape);
  const std::vector<I64> positions =
      ShapeToDims(sources[3]->GetStorage().shape);
  const std::vector<I64> output =
      ShapeToDims(destinations[0]->GetStorage().shape);
  if (input.empty() || positions.size() + 1 != input.size() ||
      output.size() != input.size()) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "op=" + op.Name() +
            ": per_head_rmsnorm_rope rank mismatch");
  }
  if (weight[0] != out_features || weight[1] != in_features ||
      norm[0] != head_dim || input.back() != in_features ||
      output.back() != out_features) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "op=" + op.Name() +
            ": per_head_rmsnorm_rope weight/norm shape mismatch");
  }
  status = CheckSamePrefix(input, output, input.size() - 1,
                           "per_head_rmsnorm_rope output prefix");
  if (!status.ok()) {
    return status;
  }
  status = CheckSamePrefix(input, positions, positions.size(),
                           "per_head_rmsnorm_rope positions shape");
  if (!status.ok()) {
    return status;
  }
  return Status::Ok();
}

// 组装融合 Q/K path 的 KernelDesc。
Status BuildPerHeadRmsNormRopeDesc(const runtime_graph::Op& op,
                                   KernelDesc* desc) {
  return BuildKernelDesc(op, "per_head_rmsnorm_rope",
                         *op.DstTensors()[0], desc);
}

}  // namespace

const OpSpec kPerHeadRmsNormRopeSpec = {
    runtime_graph::OpType::kLinearPerHeadRMSNormRope,
    "per_head_rmsnorm_rope",
    false,
    0,
    &ValidatePerHeadRmsNormRope,
    &BuildPerHeadRmsNormRopeDesc,
};

}  // namespace mobilenet
