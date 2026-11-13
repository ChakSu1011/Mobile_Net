// 模块：Runtime Graph Op 类
//
// 职责：
//   表示 graph 中的一个节点，可以是普通计算、输入、输出或 runtime 边界操作；
//   保存名称、类型、属性、输入输出 tensor 连接，以及 Engine 初始化后挂载的
//   OpEnv。
//
// 边界：
//   - 不拥有 tensor，不申请显存，不执行 shape / precision / layout 推导；
//   - 生命周期由 Graph::op_map_ 持有；
//   - tensor 连接只能由 Graph 通过 friend 权限维护；
//   - 不直接依赖 GPU/OCL 算子库，kernel 执行交给 OpEnv。
//
// 依赖：
//   runtime_graph/attr.h、runtime_graph/op_env.h、runtime_graph/op_type.h、
//   C++17 标准库。

#ifndef MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_OP_H_
#define MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "Runtime/common/status.h"
#include "runtime_graph/attr.h"
#include "runtime_graph/op_env.h"
#include "runtime_graph/op_type.h"

namespace mobilenet {
namespace runtime_graph {

class Graph;
class Tensor;

// Op 表示 graph 中的一个计算节点或边界节点。
//
// 功能：
//   记录 op 名称、类型、属性、输入输出 tensor 和运行时 OpEnv，供拓扑排序、
//   Engine 初始化和 Forward 使用。
//
// 边界：
//   - src_tensors_/dst_tensors_ 只保存裸指针，不拥有 tensor；
//   - op_env_ 由 Op 拥有，Engine Uninit 时通过 ClearOpEnv 释放；
//   - attr_ 只保存算子静态属性，不保存权重地址或 runtime buffer。
class Op {
 public:
  // 创建指定名称、类型和属性的 op。
  //
  // 参数：
  //   name：op 名称，不允许为空；
  //   op_type：op 类型；
  //   attr：op 静态属性，按值复制。
  Op(const std::string& name, OpType op_type, const AttrMap& attr);

  ~Op() = default;

  Op(const Op&) = delete;
  Op& operator=(const Op&) = delete;
  Op(Op&&) = delete;
  Op& operator=(Op&&) = delete;

  // 返回 op 名称。
  const std::string& Name() const;

  // 返回 op 类型。
  OpType Type() const;

  // 返回 op 属性。
  const AttrMap& Attrs() const;

  // 设置 op 属性。
  //
  // 参数：
  //   attr_name：属性名，不允许为空；
  //   value：属性值；同名属性会被覆盖。
  // 返回：
  //   成功返回 OK；属性名为空返回 INVALID_ARGUMENT。
  Status SetAttr(const std::string& attr_name, const AttrValue& value);

  // 返回输入 tensor 列表。
  const std::vector<Tensor*>& SrcTensors() const;

  // 返回输出 tensor 列表。
  const std::vector<Tensor*>& DstTensors() const;

  // 绑定 runtime 执行环境并接管所有权。
  //
  // 参数：
  //   op_env：Engine 初始化阶段创建的 OpEnv，不允许为空。
  // 返回：
  //   成功返回 OK；op_env 为空返回 INVALID_ARGUMENT。
  // 前置条件：
  //   Engine 初始化阶段只调用一次；若已有 OpEnv，替换会销毁旧对象，调用方
  //   不得继续持有旧 OpEnv 指针。
  Status SetOpEnv(std::unique_ptr<OpEnv> op_env);

  // 清理 runtime 执行环境。
  void ClearOpEnv();

  // 返回 runtime 执行环境；未绑定时返回 nullptr。
  OpEnv* GetOpEnv() const;

 private:
  friend class Graph;

  // 添加输入 tensor 连接。
  //
  // 参数：
  //   tensor：输入 tensor，不允许为空；生命周期由 Graph 保证。
  void AddSrcTensor(Tensor* tensor);

  // 添加输出 tensor 连接。
  //
  // 参数：
  //   tensor：输出 tensor，不允许为空；生命周期由 Graph 保证。
  void AddDstTensor(Tensor* tensor);

  std::string name_;
  OpType op_type_ = OpType::kUnknown;
  AttrMap attr_;
  std::vector<Tensor*> src_tensors_;
  std::vector<Tensor*> dst_tensors_;
  std::unique_ptr<OpEnv> op_env_;
};

}  // namespace runtime_graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_OP_H_
