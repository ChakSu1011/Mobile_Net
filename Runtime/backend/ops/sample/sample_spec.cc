// 模块：Sample 算子规格
//
// 职责：
//   定义 argmax 采样的静态契约：half logits 输入、int64 token 输出；
//   校验 method=argmax、axis=-1 与形状一致性，组装 KernelDesc。
//
// 边界：
//   - 只支持 method=argmax、axis=-1；
//   - 不支持 top-k/top-p/temperature/repetition penalty。
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

// 校验 Sample 的 attr、精度与形状。
Status ValidateSample(const runtime_graph::Op& op) {
  const auto& sources = op.SrcTensors();
  const auto& destinations = op.DstTensors();
  if (sources.size() != 1 || destinations.size() != 1) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "op=" + op.Name() + ": sample requires 1 input and 1 output");
  }
  std::string method;
  Status status = GetStringAttr(op, "method", &method);
  if (!status.ok()) {
    return status;
  }
  if (method != "argmax") {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "op=" + op.Name() +
                          ": sample only supports method=argmax");
  }
  I64 axis = -1;
  status = GetIntAttr(op, "axis", &axis);
  if (!status.ok()) {
    return status;
  }
  if (axis != -1) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "op=" + op.Name() +
                          ": sample only supports axis=-1");
  }

  const C8* op_name = op.Name().c_str();
  status = CheckTensor(*sources[0], runtime_graph::Precision::kFloat16,
                       op_name, "input[0]");
  if (!status.ok()) {
    return status;
  }
  status = CheckTensor(*destinations[0],
                       runtime_graph::Precision::kInt64, op_name,
                       "output[0]");
  if (!status.ok()) {
    return status;
  }
  const std::vector<I64> logits =
      ShapeToDims(sources[0]->GetStorage().shape);
  const std::vector<I64> output =
      ShapeToDims(destinations[0]->GetStorage().shape);
  if (logits.size() < 2 || output.size() + 1 != logits.size()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "op=" + op.Name() + ": sample rank mismatch");
  }
  for (Usize i = 0; i < output.size(); ++i) {
    if (output[i] != logits[i]) {
      return MakeStatus(
          ErrorCode::kInvalidArgument,
          "op=" + op.Name() + ": sample output shape mismatch");
    }
  }
  return Status::Ok();
}

// 组装 Sample 的 KernelDesc；精度取 logits 的 half。
Status BuildSampleDesc(const runtime_graph::Op& op, KernelDesc* desc) {
  return BuildKernelDesc(op, "sample_argmax", *op.SrcTensors()[0], desc);
}

}  // namespace

const OpSpec kSampleSpec = {
    runtime_graph::OpType::kSample,
    "sample_argmax",
    false,
    0,
    &ValidateSample,
    &BuildSampleDesc,
};

}  // namespace mobilenet
