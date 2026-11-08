// 模块：融合规则表实现
//
// 职责：
//   构造阶段一 Qwen3-0.6B Decoder Block 七段融合规则，并提供按 ID/阶段
//   的只读查询。
//
// 边界：
//   - 不包含 Kernel 实现、权重或具体张量；
//   - 规则序列必须与 Qwen3 参考构图脚本/C++ 参考构建器保持一致。

#include "graph/fusion_rule.h"

#include <utility>

namespace mobilenet {
namespace graph {
namespace {

// 构造一条规则。
FusionRule MakeRule(std::string rule_id, std::string model_family,
                    BlockStage stage, OpType op_type,
                    std::vector<std::string> required_op_kinds,
                    std::map<std::string, AttributeValue> required_attributes,
                    std::string kernel_key) {
  FusionRule rule;
  rule.rule_id = std::move(rule_id);
  rule.model_family = std::move(model_family);
  rule.stage = stage;
  rule.op_type = op_type;
  rule.required_op_kinds = std::move(required_op_kinds);
  rule.required_attributes = std::move(required_attributes);
  rule.kernel_key = std::move(kernel_key);
  return rule;
}

}  // namespace

FusionRuleTable::FusionRuleTable() : version_("1.0") {
  rules_.push_back(MakeRule("qwen3.block.norm.attn.v1", "qwen3",
                            BlockStage::kStage1AttnNorm, OpType::kRmsNorm,
                            {"rmsnorm"}, {}, "rms_norm"));
  rules_.push_back(
      MakeRule("qwen3.block.qkv.v1", "qwen3", BlockStage::kStage2Qkv,
               OpType::kQkvProjection,
               {"q_linear", "k_linear", "v_linear", "qk_rmsnorm", "rope"},
               {{"has_qk_norm", true}, {"bias", false}}, "qkv_projection"));
  rules_.push_back(MakeRule("qwen3.block.attention.v1", "qwen3",
                            BlockStage::kStage3Attention, OpType::kAttention,
                            {"kv_cache_write", "qk_matmul", "softmax",
                             "pv_matmul", "output_projection"},
                            {{"causal", true}, {"has_output_projection", true}},
                            "attention"));
  rules_.push_back(MakeRule("qwen3.block.residual.attn.v1", "qwen3",
                            BlockStage::kStage4AttnResidual,
                            OpType::kResidualAdd, {"add"}, {}, "residual_add"));
  rules_.push_back(MakeRule("qwen3.block.norm.mlp.v1", "qwen3",
                            BlockStage::kStage5MlpNorm, OpType::kRmsNorm,
                            {"rmsnorm"}, {}, "rms_norm"));
  rules_.push_back(MakeRule(
      "qwen3.block.gate_up.v1", "qwen3", BlockStage::kStage6MlpGateUp,
      OpType::kMlpGateUpAct, {"gate_linear", "up_linear", "silu", "mul"},
      {{"activation", std::string("silu")}}, "mlp_gate_up_act"));
  rules_.push_back(MakeRule("qwen3.block.down_residual.v1", "qwen3",
                            BlockStage::kStage7MlpDownResidual,
                            OpType::kMlpDownResidual, {"down_linear", "add"},
                            {}, "mlp_down_residual"));
}

const FusionRuleTable& FusionRuleTable::Instance() {
  static const FusionRuleTable instance;
  return instance;
}

const FusionRule* FusionRuleTable::Find(std::string_view rule_id) const {
  for (const FusionRule& rule : rules_) {
    if (rule.rule_id == rule_id) {
      return &rule;
    }
  }
  return nullptr;
}

const FusionRule* FusionRuleTable::FindByStage(
    std::string_view model_family, BlockStage stage,
    const std::vector<std::string>& op_kinds) const {
  for (const FusionRule& rule : rules_) {
    if (rule.model_family == model_family && rule.stage == stage &&
        rule.required_op_kinds == op_kinds) {
      return &rule;
    }
  }
  return nullptr;
}

}  // namespace graph
}  // namespace mobilenet
