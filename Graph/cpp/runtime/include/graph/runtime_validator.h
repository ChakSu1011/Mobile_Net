// 模块：RuntimeGraph 校验器
//
// 职责：
//   - Validate：把 RuntimeGraph 转换为 ValidationGraph 视图，复用 V1~V16；
//   - ValidateExecution：执行 V17，检查 Backend/Kernel/Arena/I/O 绑定。
//
// 边界：
//   - 不执行 Forward；
//   - V17 不访问设备内存内容，只检查指针/大小/类型/形状。

#ifndef MOBILENET_GRAPH_CPP_RUNTIME_INCLUDE_GRAPH_RUNTIME_VALIDATOR_H_
#define MOBILENET_GRAPH_CPP_RUNTIME_INCLUDE_GRAPH_RUNTIME_VALIDATOR_H_

#include "graph/forward_context.h"
#include "graph/runtime_graph.h"

namespace mobilenet {
namespace graph {

// RuntimeGraph 校验入口。
class RuntimeGraphValidator {
 public:
  // V1~V16 结构校验。
  //
  // 说明：
  //   RuntimeGraph 不链接模型融合规则表，因此 V13 只检查规则 ID、
  //   kernel_key 与 fused_ops 非空等结构条件，不检查规则是否在表中。
  static Status Validate(const RuntimeGraph& graph, ValidationReport* report);

  // V17 执行前校验。
  //
  // 参数：
  //   graph：已 Finalize 且完成绑定的图；
  //   context：本次 Forward 上下文；
  //   report：输出报告。
  static Status ValidateExecution(const RuntimeGraph& graph,
                                  ForwardContext* context,
                                  ValidationReport* report);
};

}  // namespace graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_CPP_RUNTIME_INCLUDE_GRAPH_RUNTIME_VALIDATOR_H_
