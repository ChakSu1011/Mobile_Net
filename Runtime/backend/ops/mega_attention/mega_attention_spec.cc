// 模块：MegaAttention 算子规格
//
// 职责：
//   定义 causal GQA attention 的静态契约：half Q/K/V 输入与 half 输出；
//   校验 head 数、GQA 比例、causal/softmax_scale/merge_heads 等属性与
//   形状一致性，组装 KernelDesc。
//
// 边界：
//   - 只支持 merge_heads=true、has_output_projection=false、
//     use_kv_cache=false；
//   - 不支持非 half 精度、KV cache 或输出投影。
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

// 校验 MegaAttention 的 attr、精度与形状。
Status ValidateMegaAttention(const runtime_graph::Op& op) {
  const auto& sources = op.SrcTensors();
  const auto& destinations = op.DstTensors();
  if (sources.size() != 3 || destinations.size() != 1) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "op=" + op.Name() +
            ": mega_attention requires 3 inputs and 1 output");
  }

  I64 num_heads = 0;
  I64 num_kv_heads = 0;
  I64 head_dim = 0;
  Bool causal = true;
  Bool merge_heads = true;
  FP64 softmax_scale = 0.0;
  Status status = GetIntAttr(op, "num_heads", &num_heads);
  if (!status.ok()) {
    return status;
  }
  status = GetIntAttr(op, "num_kv_heads", &num_kv_heads);
  if (!status.ok()) {
    return status;
  }
  status = GetIntAttr(op, "head_dim", &head_dim);
  if (!status.ok()) {
    return status;
  }
  status = GetBoolAttr(op, "causal", &causal);
  if (!status.ok()) {
    return status;
  }
  status = GetBoolAttr(op, "merge_heads", &merge_heads);
  if (!status.ok()) {
    return status;
  }
  status = GetFloatAttr(op, "softmax_scale", &softmax_scale);
  if (!status.ok()) {
    return status;
  }
  if (!merge_heads) {
    return MakeStatus(
        ErrorCode::kUnsupportedOp,
        "op=" + op.Name() +
            ": mega_attention only supports merge_heads=true");
  }
  if (HasAttr(op.Attrs(), "has_output_projection")) {
    Bool has_projection = false;
    status = GetBoolAttr(op, "has_output_projection", &has_projection);
    if (!status.ok()) {
      return status;
    }
    if (has_projection) {
      return MakeStatus(
          ErrorCode::kUnsupportedOp,
          "op=" + op.Name() +
              ": mega_attention output projection is unsupported");
    }
  }
  if (HasAttr(op.Attrs(), "use_kv_cache")) {
    Bool use_kv_cache = false;
    status = GetBoolAttr(op, "use_kv_cache", &use_kv_cache);
    if (!status.ok()) {
      return status;
    }
    if (use_kv_cache) {
      return MakeStatus(
          ErrorCode::kUnsupportedOp,
          "op=" + op.Name() +
              ": mega_attention KV cache is not supported in v1");
    }
  }
  if (HasAttr(op.Attrs(), "sliding_window")) {
    I64 sliding_window = 0;
    status = GetIntAttr(op, "sliding_window", &sliding_window);
    if (!status.ok()) {
      return status;
    }
    if (sliding_window != 0) {
      return MakeStatus(
          ErrorCode::kUnsupportedOp,
          "op=" + op.Name() +
              ": mega_attention sliding window is unsupported");
    }
  }
  if (HasAttr(op.Attrs(), "gqa_ratio")) {
    I64 gqa_ratio = 0;
    status = GetIntAttr(op, "gqa_ratio", &gqa_ratio);
    if (!status.ok()) {
      return status;
    }
    if (num_kv_heads <= 0 || gqa_ratio != num_heads / num_kv_heads) {
      return MakeStatus(
          ErrorCode::kInvalidArgument,
          "op=" + op.Name() + ": mega_attention gqa_ratio mismatch");
    }
  }
  if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
      num_heads % num_kv_heads != 0 || softmax_scale <= 0.0) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "op=" + op.Name() + ": invalid mega_attention attrs");
  }
  (void)causal;

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
  for (Usize i = 0; i < sources.size(); ++i) {
    status = CheckRank(*sources[i], 3, op_name, "input");
    if (!status.ok()) {
      return status;
    }
  }
  status = CheckRank(*destinations[0], 3, op_name, "output[0]");
  if (!status.ok()) {
    return status;
  }

  const std::vector<I64> query =
      ShapeToDims(sources[0]->GetStorage().shape);
  const std::vector<I64> key =
      ShapeToDims(sources[1]->GetStorage().shape);
  const std::vector<I64> value =
      ShapeToDims(sources[2]->GetStorage().shape);
  const std::vector<I64> output =
      ShapeToDims(destinations[0]->GetStorage().shape);
  if (query[0] != key[0] || query[0] != value[0] ||
      query[1] != key[1] || query[1] != value[1]) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "op=" + op.Name() + ": mega_attention prefix mismatch");
  }
  if (query[2] != num_heads * head_dim ||
      key[2] != num_kv_heads * head_dim ||
      value[2] != num_kv_heads * head_dim) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "op=" + op.Name() + ": mega_attention head width mismatch");
  }
  if (output != query) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "op=" + op.Name() + ": mega_attention output shape mismatch");
  }
  return Status::Ok();
}

// 组装 MegaAttention 的 KernelDesc。
Status BuildMegaAttentionDesc(const runtime_graph::Op& op,
                              KernelDesc* desc) {
  return BuildKernelDesc(op, "mega_attention", *op.DstTensors()[0], desc);
}

}  // namespace

const OpSpec kMegaAttentionSpec = {
    runtime_graph::OpType::kMegaAttention,
    "mega_attention",
    false,
    0,
    &ValidateMegaAttention,
    &BuildMegaAttentionDesc,
};

}  // namespace mobilenet
