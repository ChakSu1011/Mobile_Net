// 模块：Backend Kernel 执行上下文
//
// 职责：
//   定义 OpEnv 传给 Kernel 的单节点执行上下文：输入/输出张量视图、
//   Attention/KV 元数据、workspace、Backend/Stream 句柄与本步 token 信息。
//   Kernel 实现只通过本结构访问设备数据，不直接依赖 Graph 执行接口。
//
// 边界：
//   - 本文件只定义数据结构，不分配内存、不调用设备 API；
//   - TensorView.data 在设备侧是 Buffer*（不透明），上层不得解引用；
//   - KernelContext 由 OpEnv 在每次 Forward 内构造，生命周期仅一次 Execute；
//   - 不定义 Graph 拓扑、调度、KV 分配或采样策略；
//   - 公共头不出现 OpenCL/Vulkan/CUDA 类型。
//
// 依赖：
//   Graph/include/attr.h、shape.h、types.h；
//   Runtime/common/data_types.h、kv_view.h、sequence.h；C++17 标准库。

#ifndef MOBILENET_RUNTIME_BACKEND_KERNEL_CONTEXT_H_
#define MOBILENET_RUNTIME_BACKEND_KERNEL_CONTEXT_H_

#include <cstdint>

#include "Graph/include/attr.h"
#include "Graph/include/shape.h"
#include "Graph/include/types.h"
#include "Runtime/common/data_types.h"
#include "Runtime/common/kv_view.h"
#include "Runtime/common/sequence.h"

namespace mobilenet {

class Backend;
class Stream;

namespace runtime_graph {

// 单张量的执行视图。
//
// 功能：
//   把 Graph Storage 中的形状、精度、布局与设备地址转换为 Kernel 可直接
//   使用的形式；设备侧地址始终是后端 Buffer 句柄加字节偏移。
//
// 边界：
//   - 不拥有 data 指向的 Buffer 或主机内存；
//   - 不负责分配、释放、同步或类型转换；
//   - 仅对 row_major 连续张量有效；Kernel 必须按 shape 与 bytes 校验。
struct TensorView {
  // 设备侧为 Buffer*；主机侧为主机地址；未绑定时为空。
  void* data = nullptr;

  // data 为 Buffer 时的字节偏移；主机侧恒为 0。
  uint64_t offset = 0;

  // 有效字节数；必须等于 shape.GetShapeSize() * 元素字节数。
  uint64_t bytes = 0;

  // 逻辑形状；物理布局按 row_major 连续处理。
  Shape shape;

  // 后端边界精度；由 Storage.precision 转换得到。
  DataType dtype = DataType::kUnknown;

  // 内存布局；首期只支持 kRowMajor。
  Layout layout = Layout::kRowMajor;

  // 数据所在设备类别。
  DeviceType device = DeviceType::kUnknown;
};

// Attention 本步元数据。
//
// 功能：
//   为 Attention Kernel 提供 batch、有效 token 数、逐行序列长度以及阶段二
//   的 KV 块表/slot 信息；首期仅使用 batch/token_count/seq_lens/is_prefill。
//
// 边界：
//   - 所有指针均为非拥有，生命周期由 Executor/StepContext 保证；
//   - 不包含块分配或块表修改接口；
//   - 不表示 Prefill/Decode 的调度决策，只传递执行所需信息。
struct AttentionContext {
  // batch 行数。
  int32_t batch = 0;

  // 本步有效 token 总数。
  int32_t token_count = 0;

  // 每 batch 行的有效序列长度，长度至少为 batch。
  const int32_t* seq_lens = nullptr;

  // 每请求物理块表；首期可为空。
  const int32_t* block_table = nullptr;

  // block_table 每行的元素数。
  int32_t block_table_stride = 0;

  // KV 写入 slot；首期可为空。
  const int64_t* slots = nullptr;

  // slot 数量。
  int32_t slot_count = 0;

  // 每个 KV 块的 token 数；0 表示未启用 KV。
  int32_t kv_block_size = 0;

  // KV 只读视图；首期可为空。
  const runtime::KvCacheView* kv = nullptr;

  // 是否处于 Prefill 阶段。
  bool is_prefill = true;
};

// 单节点执行上下文。
//
// 功能：
//   把 OpEnv 缓存的张量视图与本步动态信息组合后交给 Kernel::Execute；
//   Kernel 只从本结构读取输入、输出、属性与执行参数。
//
// 边界：
//   - 由 OpEnv 在 Forward 内构造于栈上，Kernel 不得保存其指针；
//   - inputs/outputs 的元素个数必须与 op 的 src/dst tensor 数量一致；
//   - attrs 指向 Op::Attrs()，只读；
//   - 不拥有 Backend/Stream/Buffer，也不负责同步。
struct KernelContext {
  // 输入张量视图数组。
  const TensorView* inputs = nullptr;

  // 输入张量数量。
  uint32_t input_count = 0;

  // 输出张量视图数组。
  TensorView* outputs = nullptr;

  // 输出张量数量。
  uint32_t output_count = 0;

  // 节点静态属性；只读，可为空表示无属性。
  const AttrMap* attrs = nullptr;

  // Attention 元数据；非 Attention op 为空。
  const AttentionContext* attention = nullptr;

  // 采样参数；仅 Sample op 使用，首期允许为空。
  const runtime::SampleParams* sampling = nullptr;

  // 临时工作区；无 workspace 时 data 为空、bytes 为 0。
  TensorView workspace;

  // 后端实例；非拥有，可为空仅用于不访问设备的测试 Kernel。
  Backend* backend = nullptr;

  // 执行流；为空表示默认流。
  Stream* stream = nullptr;

  // 本步 batch 行数。
  int32_t batch = 1;

  // 本步有效 token 总数。
  int32_t token_count = 0;

  // 逻辑序列长度（含 padding）。
  int32_t seq_len = 0;
};

}  // namespace runtime_graph
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_KERNEL_CONTEXT_H_
