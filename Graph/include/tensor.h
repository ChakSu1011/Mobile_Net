// 模块：Runtime Graph Tensor 类
//
// 职责：
//   表示 op 之间的数据依赖边，记录 producer/consumer op、内嵌 Storage 和
//   constant 标记；Tensor 是拓扑排序、生命周期分析与显存规划的基础对象。
//
// 边界：
//   - 不拥有计算逻辑，不执行 shape / precision / layout 推导；
//   - 不拥有 Storage.data_ptr 指向的内存，也不决定显存分配策略；
//   - 生命周期由 Graph::tensor_map_ 或 Graph::constant_tensor_map_ 持有；
//   - 不对外暴露连接修改接口，连接维护只允许 Graph 调用。
//
// 依赖：
//   storage.h、C++17 标准库。

#ifndef MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_TENSOR_H_
#define MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_TENSOR_H_

#include <string>
#include <vector>

#include "storage.h"

namespace mobilenet {
namespace runtime_graph {

class Graph;
class Op;

// Tensor 表示 graph 中的数据边，并承载对应 storage 描述。
//
// 功能：
//   保存 tensor 名称、producer op 列表、consumer op 列表、Storage 和
//   constant 标记，供 Graph 查询和运行时模块消费。
//
// 边界：
//   - 普通 tensor 通常只有一个 producer op；
//   - constant tensor 没有 producer op，作为依赖叶子参与拓扑遍历；
//   - 成员函数不做线性复杂度查询，调用方应通过 Graph 接口获取 tensor。
class Tensor {
 public:
  // 创建指定名称的 tensor。
  //
  // 参数：
  //   name：tensor 名称，必须由 Graph 保证全局唯一且非空。
  explicit Tensor(const std::string& name);

  ~Tensor() = default;

  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  Tensor(Tensor&&) = delete;
  Tensor& operator=(Tensor&&) = delete;

  // 返回 tensor 名称。
  const std::string& Name() const;

  // 返回 producer op 列表。
  const std::vector<Op*>& SrcOps() const;

  // 返回 consumer op 列表。
  const std::vector<Op*>& DstOps() const;

  // 返回 tensor 的 storage。
  Storage& GetStorage();

  // 返回 tensor 的只读 storage。
  const Storage& GetStorage() const;

  // 返回 tensor storage 是否允许参与 workspace 复用。
  bool Reusable() const;

  // 设置 tensor storage 是否允许参与 workspace 复用。
  //
  // 参数：
  //   reusable：true 表示允许复用，false 表示生命周期覆盖整个 graph。
  void SetReusable(bool reusable);

  // 返回 tensor 是否为 constant 权重 tensor。
  bool IsConstant() const;

 private:
  friend class Graph;

  // 添加 producer op。
  //
  // 参数：
  //   op：产生该 tensor 的 op，不允许为空；调用方负责保证生命周期。
  void AddSrcOp(Op* op);

  // 添加 consumer op。
  //
  // 参数：
  //   op：消费该 tensor 的 op，不允许为空；调用方负责保证生命周期。
  void AddDstOp(Op* op);

  // 设置 constant 标记；仅供 Graph::CreateConstant 调用。
  //
  // 参数：
  //   is_constant：true 表示该 tensor 是权重常量。
  void SetConstant(bool is_constant);

  std::string name_;
  std::vector<Op*> src_ops_;
  std::vector<Op*> dst_ops_;
  Storage storage_;
  int ref_count_ = 0;
  bool is_constant_ = false;
};

}  // namespace runtime_graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_TENSOR_H_
