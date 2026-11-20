// 模块：Runtime KV 视图与规格
//
// 职责：
//   定义 KVCache 的规格（块大小、层数、KV 头数、头维度、精度、类型）与
//   Executor/Kernel 消费的只读视图（每层 K/V 基址与块步长）。
//
// 边界：
//   - 不定义分配算法、块池或引用计数；
//   - 视图只持有裸指针，不拥有设备内存；
//   - 块大小由后端 Attention tile 决定，不是模型常量。
//
// 依赖：
//   Runtime/common/data_types.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_COMMON_KV_VIEW_H_
#define MOBILENET_RUNTIME_COMMON_KV_VIEW_H_

#include <cstdint>
#include <vector>

#include "Runtime/common/data_types.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace runtime {

// KV Cache 类型。
//
// 说明：
//   阶段一/二只使用 kFullAttention；kSlidingWindow/kMla 为阶段三扩展，
//   当前实现必须拒绝或按全注意力处理并在日志中声明。
enum class KvCacheType {
  kFullAttention = 0,
  kSlidingWindow = 1,
  kMla = 2,
};

// KV Cache 规格。
//
// 功能：
//   描述 KVCache 初始化所需的模型与布局参数；block_size 来自后端
//   BackendCapability.attention_block_size。
//
// 边界：
//   - 不包含预算或块数（由 IKVCache::Init 根据预算计算）；
//   - dtype 使用后端边界类型 DataType；
//   - sliding_window 仅阶段三使用，全注意力为 0。
struct KvCacheSpec {
  // KV 类型。
  KvCacheType type = KvCacheType::kFullAttention;

  // 每块 token 数；由后端 Attention tile 决定。
  I32 block_size = 0;

  // 层数。
  I32 num_layers = 0;

  // KV 头数。
  I32 num_kv_heads = 0;

  // 每头维度。
  I32 head_dim = 0;

  // KV 精度。
  DataType dtype = DataType::kFloat16;

  // 滑窗大小；全注意力为 0。
  I32 sliding_window = 0;
};

// 单层 KV 视图。
//
// 功能：
//   给出该层 K/V 缓冲句柄、块步长与形状信息，Attention Kernel 用 slot
//   计算实际地址。
//
// 边界：
//   - 不拥有内存；生命周期由 IKVCache 保证；
//   - k_buffer/v_buffer 与 TensorView.data 同约定：设备侧为 Buffer*，
//     不是 Buffer::DevicePtr()；Kernel 按后端类型转换；
//   - block_stride 为单个物理块的字节数（K 或 V 单边）；
//   - 阶段一固定 NHD、块主序布局。
struct KvLayerView {
  // K 缓冲句柄（Buffer*）。
  void* k_buffer = nullptr;

  // V 缓冲句柄（Buffer*）。
  void* v_buffer = nullptr;

  // 单个物理块的字节步长。
  I64 block_stride = 0;

  // 物理块总数。
  I64 num_blocks = 0;

  // KV 头数。
  I32 num_kv_heads = 0;

  // 每头维度。
  I32 head_dim = 0;

  // KV 精度。
  DataType dtype = DataType::kUnknown;
};

// KV 只读视图。
//
// 功能：
//   由 IKVCache::BuildView 生成，Engine 传给 Executor；Executor 结合
//   PlanEntry.slots 构造 Attention 元数据。
//
// 边界：
//   - 视图在 KVCache 生命周期内有效；重新 Init 后旧视图失效；
//   - 不包含块表归属；块表随 ExecutionPlan 传递。
struct KvCacheView {
  // 每层 K/V 视图。
  std::vector<KvLayerView> layers;

  // 每块 token 数。
  I32 block_size = 0;

  // 最大 token 容量。
  I64 max_total_tokens = 0;
};

// KV 统计。
//
// 功能：
//   供 EngineStats/日志与测试断言。
struct KvCacheStats {
  // 每块 token 数。
  I32 block_size = 0;

  // 物理块总数。
  I64 total_blocks = 0;

  // 空闲块数。
  I64 free_blocks = 0;

  // 最大 token 容量。
  I64 max_total_tokens = 0;
};

}  // namespace runtime
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_COMMON_KV_VIEW_H_
