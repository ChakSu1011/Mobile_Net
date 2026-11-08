// 模块：Graph IR 校验器
//
// 职责：
//   把 Graph IR 转换为与张量表示无关的 ValidationGraph 视图，调用公共
//   V1~V16 规则，并注入 FusionRuleTable 完成 V13 规则一致性检查。
//
// 边界：
//   - 不执行运行时 arena/Kernel 校验（V17）；
//   - 不修改 Graph。
//
// 依赖：
//   graph.h、validation_common.h、fusion_rule.h（实现文件）。

#ifndef MOBILENET_GRAPH_CPP_IR_INCLUDE_GRAPH_GRAPH_VALIDATOR_H_
#define MOBILENET_GRAPH_CPP_IR_INCLUDE_GRAPH_GRAPH_VALIDATOR_H_

#include "graph/graph_common.h"

namespace mobilenet {
namespace graph {

class Graph;

// Graph IR 校验入口。
class GraphValidator {
 public:
  // 执行 Graph IR 的 V1~V16 校验。
  //
  // 参数：
  //   graph：待校验图；
  //   report：输出报告，函数开始时清空；不能为空。
  // 返回：
  //   校验通过返回 Status::Ok()；否则返回首个错误码。
  static Status Validate(const Graph& graph, ValidationReport* report);
};

}  // namespace graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_CPP_IR_INCLUDE_GRAPH_GRAPH_VALIDATOR_H_
