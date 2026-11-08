// 模块：Kernel 公开接口
//
// 职责：
//   定义 Kernel 描述、Kernel 抽象基类，以及 Kernel 执行上下文的前向声明。
//   RuntimeGraph 通过本接口按 node.kernel_key 选择并调用后端 Kernel。
//
// 边界：
//   - 本文件不定义 Backend 接口（见 Runtime/backend/backend.h）；
//   - 不定义 KernelContext 的字段（见 Graph/cpp/runtime/.../forward_context.h），
//     避免 Kernel 公共头文件依赖 RuntimeGraph 数据结构。
//
// 依赖：
//   Runtime/common/status.h、Runtime/common/data_types.h、
//   Runtime/kernel/kernel_types.h。

#ifndef MOBILENET_RUNTIME_KERNEL_KERNEL_H_
#define MOBILENET_RUNTIME_KERNEL_KERNEL_H_

#include <string>

#include "Runtime/common/data_types.h"
#include "Runtime/common/status.h"
#include "Runtime/kernel/kernel_types.h"

namespace mobilenet {
namespace graph {

// Kernel 执行上下文。
//
// 功能：
//   携带单个节点执行所需的输入/输出 TensorDesc、裸指针、属性、KV Cache、
//   Attention/Sampling 元数据与 workspace。
//
// 边界：
//   - 由 RuntimeGraph::Forward 在栈上构造，生命周期仅限一次 Execute；
//   - Kernel 不得保存其指针或内部裸指针到下一次调用；
//   - 定义见 Graph/cpp/runtime/include/graph/forward_context.h。
struct KernelContext;

}  // namespace graph

// Kernel 描述。
//
// 字段：
//   op_type：节点算子类型；
//   precision：计算精度；
//   kernel_key：后端实现键，例如 "rms_norm"。
struct KernelDesc {
  OpType op_type = OpType::kUnknown;
  DataType precision = DataType::kUnknown;
  std::string kernel_key;
};

// Kernel 抽象基类。
//
// 功能：
//   执行单个 RuntimeNode 的计算。实现方只依赖 KernelContext 中的裸指针与
//   TensorDesc，不依赖 LibTorch。
//
// 边界：
//   - 不负责 H2D/D2H 拷贝、arena 绑定或生命周期管理；
//   - 不拥有输入/输出缓冲区。
class Kernel {
 public:
  virtual ~Kernel() = default;

  // 执行一次节点计算。
  //
  // 参数：
  //   context：节点执行上下文，指针与描述在调用期间有效。
  // 返回：
  //   Status::Ok() 表示成功；失败时返回具体错误码。
  virtual Status Execute(const graph::KernelContext& context) = 0;
};

}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_KERNEL_KERNEL_H_
