// 模块：Graph IR 图对象
//
// 职责：
//   聚合节点、边、权重与模型配置，提供只读查询、校验与确定性 JSON 输出。
//   Graph 是工具链侧对象，不提供 Forward。
//
// 边界：
//   - 只读：构造完成后不能被修改；构造由 GraphBuilder 完成；
//   - 不拥有运行时 arena、Kernel 或 Backend；
//   - ToCanonicalJson 不访问张量数据，只输出元数据。
//
// 依赖：
//   graph_types.h、graph_common.h、graph_validator.h（实现文件）。

#ifndef MOBILENET_GRAPH_CPP_IR_INCLUDE_GRAPH_GRAPH_H_
#define MOBILENET_GRAPH_CPP_IR_INCLUDE_GRAPH_GRAPH_H_

#include <ATen/ATen.h>

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "graph/graph_common.h"
#include "graph/graph_types.h"

namespace mobilenet {
namespace graph {

class GraphBuilder;

// 融合算子图 IR。
//
// 功能：
//   保存一次构图结果，支持按名称查找节点/边/权重、查询节点输入输出张量、
//   执行 V1~V16 校验，以及生成用于跨语言比较的规范 JSON。
//
// 边界：
//   - 不支持增量修改；
//   - 不执行任何张量计算；
//   - 生命周期内持有 at::Tensor 引用，调用方需保证底层存储有效（通常由
//     Python torch.Tensor 或 Converter 持有）。
class Graph {
 public:
  // 图对象只能由 GraphBuilder 构造。
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;
  Graph(Graph&&) = default;
  Graph& operator=(Graph&&) = default;
  ~Graph() = default;

  // schema 主版本。
  uint32_t schema_version_major() const { return schema_version_major_; }

  // schema 次版本。
  uint32_t schema_version_minor() const { return schema_version_minor_; }

  // 融合规则集版本。
  const std::string& fusion_rules_version() const {
    return fusion_rules_version_;
  }

  // 模型配置。
  const ModelConfig& model_config() const { return model_config_; }

  // 节点列表（插入顺序）。
  const std::vector<Node>& nodes() const { return nodes_; }

  // 边列表（插入顺序）。
  const std::vector<Edge>& edges() const { return edges_; }

  // 权重列表（插入顺序）。
  const std::vector<WeightDesc>& weights() const { return weights_; }

  // 拓扑序节点 ID。
  const std::vector<NodeId>& topo_order() const { return topo_order_; }

  // 输入节点 ID 列表。
  const std::vector<NodeId>& input_nodes() const { return input_nodes_; }

  // 输出节点 ID 列表。
  const std::vector<NodeId>& output_nodes() const { return output_nodes_; }

  // 按名称查找节点。
  //
  // 参数：
  //   name：节点名。
  // 返回：
  //   节点 ID；不存在返回 kInvalidNodeId。
  NodeId FindNode(std::string_view name) const;

  // 按名称查找边。
  EdgeId FindEdge(std::string_view name) const;

  // 按 ref 查找权重。
  const WeightDesc* FindWeight(std::string_view ref) const;

  // 按 ID 获取节点。
  const Node* GetNode(NodeId id) const;

  // 按 ID 获取边。
  const Edge* GetEdge(EdgeId id) const;

  // 返回节点输入张量（共享底层存储的 at::Tensor 副本）。
  //
  // 参数：
  //   node_id：节点 ID。
  // 返回：
  //   输入张量列表；节点不存在或边不存在时返回空列表。
  std::vector<at::Tensor> NodeInputs(NodeId node_id) const;

  // 返回节点输出张量（共享底层存储的 at::Tensor 副本）。
  std::vector<at::Tensor> NodeOutputs(NodeId node_id) const;

  // 执行 V1~V16 校验。
  //
  // 参数：
  //   report：输出报告，可为空；为空时仍返回状态。
  // 返回：
  //   校验通过返回 Status::Ok()。
  Status Validate(ValidationReport* report) const;

  // 生成规范 JSON。
  //
  // 说明：
  //   字段顺序固定；不包含 TensorDesc/Runtime 字段；浮点使用最短往返格式；
  //   同一图多次调用结果逐字节一致。
  std::string ToCanonicalJson() const;

 private:
  friend class GraphBuilder;

  Graph() = default;

  uint32_t schema_version_major_ = 1;
  uint32_t schema_version_minor_ = 0;
  std::string fusion_rules_version_;
  ModelConfig model_config_;
  std::vector<Node> nodes_;
  std::vector<Edge> edges_;
  std::vector<WeightDesc> weights_;
  std::vector<NodeId> topo_order_;
  std::vector<NodeId> input_nodes_;
  std::vector<NodeId> output_nodes_;
  std::map<std::string, NodeId> node_ids_;
  std::map<std::string, EdgeId> edge_ids_;
  std::map<std::string, size_t> weight_indices_;
};

}  // namespace graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_CPP_IR_INCLUDE_GRAPH_GRAPH_H_
