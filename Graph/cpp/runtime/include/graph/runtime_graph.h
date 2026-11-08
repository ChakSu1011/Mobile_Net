// 模块：RuntimeGraph 图对象
//
// 职责：
//   保存运行时图结构、Backend/Kernel/Arena 绑定，提供 V1~V17 校验、
//   Forward、Release 与规范 JSON 输出。
//
// 边界：
//   - 不链接 LibTorch；只使用 TensorDesc、裸指针与 POD；
//   - 不拥有 Backend、Kernel、Arena 与 I/O 缓冲区；
//   - Release 只销毁 Kernel，不释放调用方内存。
//
// 依赖：
//   runtime_graph_types.h、forward_context.h、Runtime/backend/backend.h。

#ifndef MOBILENET_GRAPH_CPP_RUNTIME_INCLUDE_GRAPH_RUNTIME_GRAPH_H_
#define MOBILENET_GRAPH_CPP_RUNTIME_INCLUDE_GRAPH_RUNTIME_GRAPH_H_

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "Runtime/backend/backend.h"
#include "graph/forward_context.h"
#include "graph/runtime_graph_types.h"

namespace mobilenet {
namespace graph {

class RuntimeGraphBuilder;
class RuntimeGraphValidator;

// 运行时融合算子图。
//
// 生命周期：
//   1. RuntimeGraphBuilder::Finalize 构造；
//   2. Validate() 做结构校验；
//   3. BindBackend/BindArena(s)/BindKernel；
//   4. ValidateExecution()；
//   5. Forward()（可多次）；
//   6. Release() 销毁 Kernel；对象析构前必须 Release。
class RuntimeGraph {
 public:
  RuntimeGraph(const RuntimeGraph&) = delete;
  RuntimeGraph& operator=(const RuntimeGraph&) = delete;
  RuntimeGraph(RuntimeGraph&&) = default;
  RuntimeGraph& operator=(RuntimeGraph&&) = default;
  ~RuntimeGraph();

  // 结构版本。
  uint32_t schema_version_major() const { return schema_version_major_; }
  uint32_t schema_version_minor() const { return schema_version_minor_; }
  const std::string& fusion_rules_version() const {
    return fusion_rules_version_;
  }
  const RuntimeModelConfig& model_config() const { return model_config_; }

  // 节点/边/权重/拓扑序。
  const std::vector<RuntimeNode>& nodes() const { return nodes_; }
  const std::vector<RuntimeEdge>& edges() const { return edges_; }
  const std::vector<RuntimeWeightDesc>& weights() const { return weights_; }
  const std::vector<NodeId>& topo_order() const { return topo_order_; }
  const std::vector<NodeId>& input_nodes() const { return input_nodes_; }
  const std::vector<NodeId>& output_nodes() const { return output_nodes_; }

  // 按名称查找。
  NodeId FindNode(std::string_view name) const;
  EdgeId FindEdge(std::string_view name) const;

  // 按 ID 获取。
  const RuntimeNode* GetNode(NodeId id) const;
  const RuntimeEdge* GetEdge(EdgeId id) const;
  const RuntimeWeightDesc* GetWeight(std::string_view ref) const;

  // 是否已 Finalize / 已 Release。
  bool finalized() const { return finalized_; }
  bool released() const { return released_; }

  // V1~V16 结构校验。
  Status Validate(ValidationReport* report) const;

  // 绑定 Backend；只能绑定一次。
  Status BindBackend(Backend* backend, ValidationReport* report);

  // 绑定单个 Arena；参数非法或重复绑定时失败。
  Status BindArena(int32_t arena_id, void* base, uint64_t bytes,
                   ValidationReport* report);

  // 原子绑定多个 Arena：全部合法才提交。
  Status BindArenas(const std::map<int32_t, ArenaBinding>& arenas,
                    ValidationReport* report);

  // 为计算节点绑定 Kernel。
  Status BindKernel(NodeId node_id, Kernel* kernel, ValidationReport* report);

  // V17：执行前校验 Backend/Kernel/Arena/I/O 绑定。
  Status ValidateExecution(ForwardContext* context,
                           ValidationReport* report) const;

  // 执行一次 Forward。
  //
  // 参数：
  //   context：I/O 绑定、KV Cache、Attention、Sampling 与 workspace；
  //   report：失败时写入问题。
  // 返回：
  //   Status::Ok() 或具体错误。
  Status Forward(ForwardContext* context, ValidationReport* report);

  // 逆拓扑序销毁 Kernel；可重复调用。
  Status Release(ValidationReport* report);

  // 规范 JSON；字段顺序与 Graph IR 一致。
  std::string ToCanonicalJson() const;

 private:
  friend class RuntimeGraphBuilder;
  friend class RuntimeGraphValidator;

  RuntimeGraph() = default;

  // 单个节点的栈上 I/O 视图；避免 Forward 稳态分配。
  static constexpr size_t kMaxNodePorts = 32;
  struct KernelIo {
    std::array<const TensorDesc*, kMaxNodePorts> input_tensors{};
    std::array<const TensorDesc*, kMaxNodePorts> output_tensors{};
    std::array<void*, kMaxNodePorts> input_ptrs{};
    std::array<void*, kMaxNodePorts> output_ptrs{};
  };

  // 解析边指针并构造 KernelContext。
  Status BuildKernelContext(NodeId node_id, ForwardContext* context,
                            KernelIo* io, KernelContext* kernel_context) const;

  bool finalized_ = false;
  bool released_ = false;
  uint32_t schema_version_major_ = 1;
  uint32_t schema_version_minor_ = 0;
  std::string fusion_rules_version_;
  RuntimeModelConfig model_config_;
  std::vector<RuntimeNode> nodes_;
  std::vector<RuntimeEdge> edges_;
  std::vector<RuntimeWeightDesc> weights_;
  std::vector<NodeId> topo_order_;
  std::vector<NodeId> input_nodes_;
  std::vector<NodeId> output_nodes_;
  std::map<std::string, NodeId> node_ids_;
  std::map<std::string, EdgeId> edge_ids_;
  std::map<std::string, size_t> weight_indices_;
  Backend* backend_ = nullptr;
  std::map<int32_t, ArenaBinding> arenas_;
  std::vector<Kernel*> node_kernels_;
};

}  // namespace graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_CPP_RUNTIME_INCLUDE_GRAPH_RUNTIME_GRAPH_H_
