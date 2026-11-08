// 模块：RuntimeGraph 构建器
//
// 职责：
//   增量收集 RuntimeNode/RuntimeEdge/RuntimeWeightDesc，Finalize 时解析边
//   引用、计算拓扑序并执行 V1~V16 结构校验。
//
// 边界：
//   - 不分配 arena、不创建 Kernel、不做 H2D/D2H；
//   - Finalize 失败时 Builder 保持可用；
//   - 同一 Builder 只能成功 Finalize 一次。

#ifndef MOBILENET_GRAPH_CPP_RUNTIME_INCLUDE_GRAPH_RUNTIME_GRAPH_BUILDER_H_
#define MOBILENET_GRAPH_CPP_RUNTIME_INCLUDE_GRAPH_RUNTIME_GRAPH_BUILDER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "graph/runtime_graph.h"
#include "graph/runtime_graph_types.h"

namespace mobilenet {
namespace graph {

// RuntimeGraph 构建器。
class RuntimeGraphBuilder {
 public:
  RuntimeGraphBuilder();
  ~RuntimeGraphBuilder() = default;

  RuntimeGraphBuilder(const RuntimeGraphBuilder&) = delete;
  RuntimeGraphBuilder& operator=(const RuntimeGraphBuilder&) = delete;

  Status AddNode(RuntimeNode node, NodeId* node_id);
  Status AddEdge(RuntimeEdge edge, EdgeId* edge_id);
  Status AddWeight(RuntimeWeightDesc weight);
  Status SetModelConfig(RuntimeModelConfig model_config);
  Status SetSchemaVersion(uint32_t major, uint32_t minor);
  Status SetFusionRulesVersion(std::string version);
  Status SetInputNodes(std::vector<NodeId> node_ids);
  Status SetOutputNodes(std::vector<NodeId> node_ids);

  // 完成构图。
  //
  // 参数：
  //   graph：输出 RuntimeGraph；不能为空。
  // 返回：
  //   Status::Ok()；失败时 graph 未修改。
  Status Finalize(std::unique_ptr<RuntimeGraph>* graph);

  const ValidationReport& validation_report() const {
    return validation_report_;
  }

 private:
  Status Fail(ErrorCode code, std::string message);

  std::vector<RuntimeNode> nodes_;
  std::vector<RuntimeEdge> edges_;
  std::vector<RuntimeWeightDesc> weights_;
  std::map<std::string, NodeId> node_ids_;
  std::map<std::string, EdgeId> edge_ids_;
  std::set<std::string> weight_refs_;
  std::vector<NodeId> input_nodes_;
  std::vector<NodeId> output_nodes_;
  RuntimeModelConfig model_config_;
  uint32_t schema_version_major_ = 1;
  uint32_t schema_version_minor_ = 0;
  std::string fusion_rules_version_;
  ValidationReport validation_report_;
  bool finalized_ = false;
};

}  // namespace graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_CPP_RUNTIME_INCLUDE_GRAPH_RUNTIME_GRAPH_BUILDER_H_
