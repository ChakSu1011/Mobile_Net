// 模块：融合规则表
//
// 职责：
//   定义模型无关的 FusionRule 结构与按 model_family 索引的规则表；
//   阶段一首个规则集为 Qwen3-0.6B 的 Decoder Block 七段融合。
//
// 边界：
//   - 规则表只描述“什么算子序列可以融合成什么节点”，不包含 Kernel 实现；
//   - 不包含具体模型权重或张量；
//   - 其他模型族通过新增规则集扩展，不修改 Graph 对象模型。
//
// 依赖：
//   graph_common.h、Runtime/kernel/kernel_types.h。

#ifndef MOBILENET_GRAPH_CPP_IR_INCLUDE_GRAPH_FUSION_RULE_H_
#define MOBILENET_GRAPH_CPP_IR_INCLUDE_GRAPH_FUSION_RULE_H_

#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "graph/graph_common.h"

namespace mobilenet {
namespace graph {

// 单条融合规则。
//
// 字段：
//   rule_id：全局唯一规则 ID；
//   model_family：适用模型族；
//   stage：Decoder Block 融合阶段；
//   op_type：融合后节点类型；
//   required_op_kinds：融合前算子序列（按顺序精确匹配）；
//   required_attributes：融合后节点必须具备的属性及取值；
//   kernel_key：推荐后端实现键；为空表示由转换器决定。
struct FusionRule {
  std::string rule_id;
  std::string model_family;
  BlockStage stage = BlockStage::kUnknown;
  OpType op_type = OpType::kUnknown;
  std::vector<std::string> required_op_kinds;
  std::map<std::string, AttributeValue> required_attributes;
  std::string kernel_key;
};

// 融合规则表单例。
//
// 功能：
//   按 rule_id 或 (model_family, stage, op_kinds) 查询规则。
//
// 边界：
//   - 规则集在首次 Instance() 时构造，之后只读；
//   - 当前包含 Qwen3 规则集；新增模型族必须保持 rule_id 唯一。
class FusionRuleTable {
 public:
  // 获取全局单例。
  static const FusionRuleTable& Instance();

  // 规则集版本。
  const std::string& version() const { return version_; }

  // 全部规则。
  const std::vector<FusionRule>& rules() const { return rules_; }

  // 按 rule_id 查找。
  //
  // 参数：
  //   rule_id：规则 ID。
  // 返回：
  //   规则指针；不存在返回 nullptr。
  const FusionRule* Find(std::string_view rule_id) const;

  // 按模型族、阶段与算子序列查找。
  //
  // 参数：
  //   model_family：模型族；
  //   stage：融合阶段；
  //   op_kinds：融合前算子序列。
  // 返回：
  //   规则指针；不存在返回 nullptr。
  const FusionRule* FindByStage(std::string_view model_family, BlockStage stage,
                                const std::vector<std::string>& op_kinds) const;

  FusionRuleTable(const FusionRuleTable&) = delete;
  FusionRuleTable& operator=(const FusionRuleTable&) = delete;

 private:
  FusionRuleTable();

  std::string version_;
  std::vector<FusionRule> rules_;
};

}  // namespace graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_CPP_IR_INCLUDE_GRAPH_FUSION_RULE_H_
