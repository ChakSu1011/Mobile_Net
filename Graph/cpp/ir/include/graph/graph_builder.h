// 模块：Graph IR 构建器
//
// 职责：
//   以增量方式收集节点、边、权重与模型配置，在 Finalize() 时解析边引用、
//   计算确定性拓扑序、执行 V1~V16 校验并产出只读 Graph。
//
// 边界：
//   - 不执行张量计算、不分配运行时 arena；
//   - Finalize 失败时 Builder 保持可用，调用方可修正后重试；
//   - 同一 Builder 只能成功 Finalize 一次。
//
// 依赖：
//   graph.h、graph_types.h、graph_validator.h。

#ifndef MOBILENET_GRAPH_CPP_IR_INCLUDE_GRAPH_GRAPH_BUILDER_H_
#define MOBILENET_GRAPH_CPP_IR_INCLUDE_GRAPH_GRAPH_BUILDER_H_

#include <ATen/ATen.h>

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "graph/graph.h"
#include "graph/graph_common.h"
#include "graph/graph_types.h"

namespace mobilenet {
namespace graph {

// Graph IR 构建器。
class GraphBuilder {
 public:
  GraphBuilder();
  ~GraphBuilder() = default;

  GraphBuilder(const GraphBuilder&) = delete;
  GraphBuilder& operator=(const GraphBuilder&) = delete;

  // 添加节点。
  //
  // 参数：
  //   node：节点；name 必须合法且唯一。
  //   node_id：输出节点 ID。
  // 返回：
  //   Status::Ok() 或 INVALID_ARGUMENT/STATE_ERROR。
  Status AddNode(Node node, NodeId* node_id);

  // 添加边。
  //
  // 参数：
  //   edge：边；name 唯一，tensor 必须已定义，symbolic_shape 必须与
  //         tensor.sizes() 一致。
  //   edge_id：输出边 ID。
  // 返回：
  //   Status::Ok() 或 INVALID_ARGUMENT。
  Status AddEdge(Edge edge, EdgeId* edge_id);

  // 添加权重。
  //
  // 参数：
  //   weight：权重；ref 唯一且 tensor 已定义。
  // 返回：
  //   Status::Ok() 或 INVALID_ARGUMENT。
  Status AddWeight(WeightDesc weight);

  // 设置模型配置。
  Status SetModelConfig(ModelConfig model_config);

  // 设置 schema 版本。
  Status SetSchemaVersion(uint32_t major, uint32_t minor);

  // 设置融合规则集版本。
  Status SetFusionRulesVersion(std::string version);

  // 设置输入节点集合。
  //
  // 参数：
  //   node_ids：输入节点 ID；每个 ID 必须存在且 op_type == kInput。
  Status SetInputNodes(std::vector<NodeId> node_ids);

  // 设置输出节点集合。
  Status SetOutputNodes(std::vector<NodeId> node_ids);

  // 追加输入节点。
  Status AddInputNode(NodeId node_id);

  // 追加输出节点。
  Status AddOutputNode(NodeId node_id);

  // 完成构图。
  //
  // 行为：
  //   1. 解析边 producer/consumers；
  //   2. 检查每条非权重边恰好一个 producer、至少一个 consumer；
  //   3. 用 Kahn 算法生成以 NodeId 为稳定优先级的拓扑序；
  //   4. 构建 Graph 并执行 V1~V16 校验；
  //   5. 成功后 Builder 标记为已 Finalize。
  //
  // 参数：
  //   graph：输出图；不能为空；成功时写入新 Graph。
  // 返回：
  //   Status::Ok()；失败时 graph 保持未修改，Builder 可继续修改。
  Status Finalize(std::unique_ptr<Graph>* graph);

  // 最近一次 Finalize 的校验报告。
  const ValidationReport& validation_report() const {
    return validation_report_;
  }

 private:
  // 记录添加阶段错误。
  Status Fail(ErrorCode code, std::string message);

  std::vector<Node> nodes_;
  std::vector<Edge> edges_;
  std::vector<WeightDesc> weights_;
  std::map<std::string, NodeId> node_ids_;
  std::map<std::string, EdgeId> edge_ids_;
  std::set<std::string> weight_refs_;
  std::vector<NodeId> input_nodes_;
  std::vector<NodeId> output_nodes_;
  ModelConfig model_config_;
  uint32_t schema_version_major_ = 1;
  uint32_t schema_version_minor_ = 0;
  std::string fusion_rules_version_;
  ValidationReport validation_report_;
  bool finalized_ = false;
};

}  // namespace graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_CPP_IR_INCLUDE_GRAPH_GRAPH_BUILDER_H_
