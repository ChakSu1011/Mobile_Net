// 模块：Kernel 公开接口
//
// 职责：
//   定义 Kernel 描述、Kernel 抽象基类，以及 Kernel 执行上下文的前向声明。
//   Graph/OpEnv 通过本接口按 op 类型选择并调用后端 Kernel。
//
// 边界：
//   - 本文件不定义 Backend 接口（见 Runtime/backend/backend.h）；
//   - 不定义 KernelContext 的字段（由 Executor 侧定义并构造），避免 Kernel
//     公共头文件依赖 Graph 数据结构；
//   - 算子类型统一使用 Graph 的 runtime_graph::OpType，不再使用
//     kernel_types.h 中的遗留枚举。
//
// 依赖：
//   Runtime/common/status.h、Runtime/common/data_types.h、
//   Runtime/kernel/kernel_types.h、Graph/include/op_type.h。

#ifndef MOBILENET_RUNTIME_KERNEL_KERNEL_H_
#define MOBILENET_RUNTIME_KERNEL_KERNEL_H_

#include <string>

#include "Graph/include/op_type.h"
#include "Runtime/common/data_types.h"
#include "Runtime/common/status.h"
#include "Runtime/kernel/kernel_types.h"

namespace mobilenet {

namespace runtime_graph {

// Kernel 执行上下文。
//
// 功能：
//   携带单个节点执行所需的输入/输出 TensorDesc、裸指针、属性、KV Cache、
//   Attention/Sampling 元数据与 workspace。
//
// 边界：
//   - 由 Executor 在调用 OpEnv::Forward 前构造，生命周期仅限一次 Execute；
//   - Kernel 不得保存其指针或内部裸指针到下一次调用；
//   - 具体字段由 Executor 侧头文件定义（ExecBackend 维护）。
struct KernelContext;

}  // namespace runtime_graph

// Kernel 描述。
//
// 字段：
//   op_type：图节点算子类型，必须来自 runtime_graph::OpType；
//   precision：计算精度；
//   kernel_key：后端实现键，例如 "rms_norm"。
struct KernelDesc {
  // 图节点算子类型。
  runtime_graph::OpType op_type = runtime_graph::OpType::kUnknown;

  // 计算精度。
  DataType precision = DataType::kUnknown;

  // 后端实现键。
  std::string kernel_key;
};

// Kernel 抽象基类。
//
// 功能：
//   执行单个图节点的计算。实现方只依赖 KernelContext 中的裸指针与张量描述，
//   不依赖 LibTorch。
//
// 边界：
//   - 不负责 H2D/D2H 拷贝、arena 绑定或生命周期管理；
//   - 不拥有输入/输出缓冲区；
//   - 不感知调度、请求或 KV 分配策略。
class Kernel {
 public:
  virtual ~Kernel() = default;

  // 执行一次节点计算。
  //
  // 参数：
  //   context：节点执行上下文，指针与描述在调用期间有效。
  // 返回：
  //   成功返回 kOk；失败返回具体错误码（kKernelError/kDeviceError 等）。
  virtual Status Execute(const runtime_graph::KernelContext& context) = 0;
};

}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_KERNEL_KERNEL_H_
