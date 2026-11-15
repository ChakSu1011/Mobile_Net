// 模块：Runtime KVCache 统一接口
//
// 职责：
//   定义 Engine 调用的 IKVCache：按后端 Attention tile 决定块大小、按预算
//   计算容量、分配/回收物理块、维护块表与 slot，并向 Executor 提供只读视图。
//
// 边界：
//   - 不决定抢占对象、不实现 Attention 计算；
//   - 不直接调用设备 API（经 Backend 创建/释放缓冲）；
//   - 块大小不是模型常量，必须来自 BackendCapability。
//
// 依赖：
//   Runtime/common/kv_allocator.h、kv_view.h、resource_stack.h、
//   Runtime/backend/backend.h。

#ifndef MOBILENET_RUNTIME_KVCACHE_KV_CACHE_MANAGER_H_
#define MOBILENET_RUNTIME_KVCACHE_KV_CACHE_MANAGER_H_

#include <cstdint>
#include <memory>

#include "Runtime/common/kv_allocator.h"
#include "Runtime/common/kv_view.h"
#include "Runtime/common/resource_stack.h"
#include "Runtime/common/status.h"

namespace mobilenet {

class Backend;

namespace runtime {

// KVCache 统一接口。
//
// 功能：
//   Engine 在 Load 阶段调用 Init，在每步调用 BuildView，在请求结束时调用
//   Release（继承自 KvAllocator）；Scheduler 通过 KvAllocator 子接口申请块。
//
// 边界：
//   - 只允许单推理线程使用；
//   - Init 失败时必须通过 ResourceStack 回滚已创建的缓冲；
//   - 阶段二前缀共享、阶段三分层存储均在本接口之上扩展。
class IKVCache : public KvAllocator {
 public:
  // 初始化 KV 池。
  //
  // 参数：
  //   backend：后端实例，用于创建 KV 缓冲，非空；只借用，生命周期由
  //            Executor 保证；
  //   spec：块大小/层数/KV 头数/头维度/精度；
  //   kv_budget_bytes：KV 预算（字节）；
  //   resource_stack：缓冲释放登记栈，非空。
  // 返回：
  //   成功返回 kOk；预算不足或 block_size 非法返回 kDeviceOom/
  //   kInvalidArgument；后端分配失败返回 kBufferError。
  virtual Status Init(Backend* backend, const KvCacheSpec& spec,
                      uint64_t kv_budget_bytes,
                      ResourceStack* resource_stack) = 0;

  // 构建只读视图。
  //
  // 参数：
  //   view：输出参数；写入每层 K/V 基址、块步长、块大小与容量。
  // 返回：
  //   成功返回 kOk；未初始化返回 kStateError。
  virtual Status BuildView(KvCacheView* view) const = 0;

  // 返回 KV 统计。
  virtual KvCacheStats Stats() const = 0;
};

// 创建默认 KVCache。
//
// 返回：
//   非空 KVCache；由 Engine 持有。实现方为 RuntimeCore。
std::unique_ptr<IKVCache> CreateDefaultKVCache();

}  // namespace runtime
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_KVCACHE_KV_CACHE_MANAGER_H_
