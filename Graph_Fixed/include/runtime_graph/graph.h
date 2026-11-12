// 模块：Runtime Graph 图对象
//
// 职责：
//   作为 runtime 静态图 IR 的根对象，持有 op、普通 tensor、constant tensor、
//   图输入输出、拓扑排序结果和 graph 级共享环境；提供构图、图边界注册、
//   拓扑排序与只读查询接口。
//
// 边界：
//   - 只表达“图是什么”和“对象如何连接”，不执行底层 kernel；
//   - 不负责显存申请、通信初始化或具体 shape / precision / layout 推导；
//   - 不直接依赖 GPU/OCL 算子库；
//   - constant tensor 由 constant_tensor_map_ 独立持有，不写入 tensor_map_；
//   - 拓扑排序后 op_list_ 只保存非拥有裸指针，不负责释放 op。
//
// 依赖：
//   runtime_graph/attr.h、graph_env.h、op.h、op_type.h、tensor.h、
//   Runtime/common/status.h、C++17 标准库。

#ifndef MOBILENET_GRAPH_FIXED_INCLUDE_RUNTIME_GRAPH_GRAPH_H_
#define MOBILENET_GRAPH_FIXED_INCLUDE_RUNTIME_GRAPH_GRAPH_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Runtime/common/status.h"
#include "runtime_graph/attr.h"
#include "runtime_graph/graph_env.h"
#include "runtime_graph/op.h"
#include "runtime_graph/op_type.h"
#include "runtime_graph/tensor.h"

namespace mobilenet {
namespace runtime_graph {

// Graph 是 runtime IR 的核心类，负责构图、图边界注册、拓扑排序和 IR 查询。
//
// 功能：
//   管理 op/tensor/constant 生命周期，维护图输入输出边界，执行 post DFS
//   拓扑排序，并向 Engine、Finalizer、Manager 暴露稳定查询接口。
//
// 边界：
//   - 所有 Create* 接口在失败时不修改 graph 状态（参数校验先于提交）；
//   - SortOp 以图输出为起点，constant tensor 作为依赖叶子，不进入执行序列；
//   - op_list_、input_op_list_、output_op_list_、constant_tensor_list_ 只保存
//     非拥有裸指针；
//   - 不保证线程安全，构图和排序应在单线程完成。
class Graph {
 public:
  // 创建指定名称的 graph。
  //
  // 参数：
  //   name：graph 名称，用于调试和 graph_map 索引；允许为空，空名称不做
  //         语义校验。
  explicit Graph(const std::string& name);

  // 析构 graph。
  //
  // 行为：
  //   先对所有 op 调用 ClearOpEnv()，确保 OpEnv 在 GraphEnv 之前析构；随后
  //   由成员逆序析构释放 GraphEnv、Tensor、Op 等对象，避免派生 OpEnv 在
  //   析构阶段访问已经释放的 GraphEnv。
  virtual ~Graph();

  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;
  Graph(Graph&&) = delete;
  Graph& operator=(Graph&&) = delete;

  // 返回 graph 名称。
  virtual const std::string& Name() const;

  // 创建普通计算 op，并返回该 op 创建出的输出 tensor。
  //
  // 参数：
  //   name：op 名称，必须非空且在 graph 内唯一；
  //   src_tensors：输入 tensor 列表，元素必须是当前 graph 持有的 tensor；
  //   out_cnt：输出 tensor 数量，必须大于 0；
  //   op_type：op 类型，不得为 kInput/kOutput/kUnknown；
  //   attr：op 属性，按值复制；
  //   outputs：输出参数，返回新创建的输出 tensor，调用方不拥有所有权。
  // 返回：
  //   成功返回 OK；参数非法、名称重复或 tensor 不属于当前 graph 时返回
  //   INVALID_ARGUMENT。
  virtual Status CreateOp(const std::string& name,
                          const std::vector<Tensor*>& src_tensors,
                          size_t out_cnt, OpType op_type, const AttrMap& attr,
                          std::vector<Tensor*>* outputs);

  // 创建 constant tensor，并把传入 storage 保存到 tensor 内。
  //
  // 参数：
  //   name：constant 名称，必须非空且在 graph 内唯一；
  //   storage：权重 storage 描述，data_ptr 通常指向 Python torch tensor 的
  //             CPU 权重；函数内会把 reusable 置为 false；
  //   output：输出参数，返回 constant tensor，调用方不拥有所有权。
  // 返回：
  //   成功返回 OK；名称为空/重复、storage shape 非法或 output 为空时返回
  //   INVALID_ARGUMENT。
  virtual Status CreateConstant(const std::string& name,
                                const Storage& storage, Tensor** output);

  // 创建 graph 输入 op（legacy 重载）。
  //
  // 参数：
  //   name：输入名称，同时作为 input op 和输出 tensor 的名称；
  //   output：输出参数，返回该输入 tensor，调用方不拥有所有权。
  // 返回：
  //   成功返回 OK；名称为空/重复或 output 为空时返回 INVALID_ARGUMENT。
  virtual Status CreateInput(const std::string& name, Tensor** output);

  // 创建 graph 输入 op，并显式声明构图期 precision/layout。
  //
  // 参数：
  //   name：输入名称，同时作为 input op 和输出 tensor 的名称；
  //   precision：构图期输入精度，写入 input op 的 attr["storage"]；
  //   layout：构图期输入布局，写入 input op 的 attr["storage"]；
  //   output：输出参数，返回该输入 tensor，调用方不拥有所有权。
  // 返回：
  //   成功返回 OK；参数非法或名称重复时返回 INVALID_ARGUMENT。
  virtual Status CreateInput(const std::string& name, Precision precision,
                             Layout layout, Tensor** output);

  // 创建 graph 输出 op。
  //
  // 参数：
  //   name：输出 op 名称，必须非空且在 graph 内唯一；
  //   input：输出依赖的 tensor，必须是当前 graph 持有的 tensor。
  // 返回：
  //   成功返回 OK；参数非法、名称重复或 tensor 不属于当前 graph 时返回
  //   INVALID_ARGUMENT。
  virtual Status CreateOutput(const std::string& name, Tensor* input);

  // 根据 graph 输出依赖整理 op 列表。
  //
  // 功能：
  //   从 output_op_list_ 出发，沿 tensor 的 producer 依赖执行 post DFS，
  //   将 op 按拓扑顺序写入 op_list_；constant tensor 作为依赖叶子，不压入
  //   producer op。
  //
  // 返回：
  //   成功返回 OK；图没有输出、依赖环、非 constant tensor 没有 producer 或
  //   tensor 连接为空时返回 STATE_ERROR。
  virtual Status SortOp();

  // 返回所有 constant tensor。
  virtual const std::vector<Tensor*>& GetAllConstantTensor();

  // 返回所有输入 op。
  virtual const std::vector<Op*>& GetAllInputOp();

  // 返回所有输出 op。
  virtual const std::vector<Op*>& GetAllOutputOp();

  // 返回拓扑排序后的执行 op 序列。
  virtual const std::vector<Op*>& GetOrderlyOPList();

  // 设置 graph 级共享环境资源，并接管所有权。
  //
  // 参数：
  //   graph_env：Engine 创建出的 GraphEnv，不允许为空。
  // 返回：
  //   成功返回 OK；graph_env 为空返回 INVALID_ARGUMENT。
  // 前置条件：
  //   Engine 初始化阶段只调用一次；若已有 GraphEnv，调用方必须先清理所有
  //   OpEnv，否则既有 OpEnv 保存的 graph_env_ 指针会悬空。
  virtual Status SetGraphEnv(std::unique_ptr<GraphEnv> graph_env);

  // 返回 graph 级共享环境资源；未创建时返回 nullptr。
  virtual GraphEnv* GetGraphEnv();

  // 清理 graph 级共享环境资源。
  //
  // 前置条件：
  //   所有 OpEnv 必须已经通过 ClearOpEnv 释放，避免悬空 graph_env_ 指针。
  virtual void ClearGraphEnv();

 protected:
  // 检查 tensor 指针是否由当前 graph 持有。
  //
  // 参数：
  //   tensor：待检查的 tensor 指针，允许为空。
  // 返回：
  //   true 表示该指针等于 tensor_map_ 或 constant_tensor_map_ 中某个对象的
  //   get() 结果；空指针返回 false。
  bool OwnsTensor(const Tensor* tensor) const;

  std::string name_;
  std::vector<Op*> op_list_;
  std::vector<Tensor*> constant_tensor_list_;
  std::vector<Op*> input_op_list_;
  std::vector<Op*> output_op_list_;
  std::map<std::string, std::unique_ptr<Op>> op_map_;
  std::map<std::string, std::unique_ptr<Tensor>> tensor_map_;
  std::map<std::string, std::unique_ptr<Tensor>> constant_tensor_map_;
  std::unique_ptr<GraphEnv> graph_env_;

 private:
  // 校验新 op 名称。
  //
  // 参数：
  //   name：待校验名称。
  // 返回：
  //   名称非空且未被占用时返回 OK，否则返回 INVALID_ARGUMENT。
  Status ValidateNewOpName(const std::string& name) const;

  // 校验新普通 tensor 名称。
  //
  // 参数：
  //   name：待校验名称。
  // 返回：
  //   名称非空且未被 tensor_map_/constant_tensor_map_ 占用时返回 OK，
  //   否则返回 INVALID_ARGUMENT。
  Status ValidateNewTensorName(const std::string& name) const;

  // 创建输入 op 和输入 tensor 的公共实现。
  //
  // 参数：
  //   name：输入名称；
  //   attr_storage：写入 input op attr["storage"] 的 storage 声明；
  //   output：输出参数，返回输入 tensor。
  // 返回：
  //   成功返回 OK，否则返回 INVALID_ARGUMENT。
  Status CreateInputInternal(const std::string& name,
                             const Storage& attr_storage, Tensor** output);

  // 对一个 op 执行 post DFS。
  //
  // 参数：
  //   op：当前访问的 op；
  //   states：访问状态表，0=未访问，1=访问中，2=已完成。
  // 返回：
  //   成功返回 OK；发现依赖环或缺失 producer 时返回 STATE_ERROR。
  Status VisitOp(Op* op, std::map<const Op*, int>* states);
};

}  // namespace runtime_graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_FIXED_INCLUDE_RUNTIME_GRAPH_GRAPH_H_
