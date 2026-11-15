// 模块：Runtime KV 分配接口
//
// 职责：
//   定义 Scheduler 视角的 KV 分配统一接口：按请求追加分配、释放、查询空闲
//   块、读取块表与 slot；具体实现由 KVCache 模块提供。
//
// 边界：
//   - 不暴露物理布局、层缓冲或块池内部结构；
//   - 不负责选择抢占对象（由 Scheduler 决定）；
//   - Reserve 失败必须是零副作用，调用方可安全重试。
//
// 依赖：
//   Runtime/common/status.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_COMMON_KV_ALLOCATOR_H_
#define MOBILENET_RUNTIME_COMMON_KV_ALLOCATOR_H_

#include <cstdint>
#include <vector>

#include "Runtime/common/status.h"

namespace mobilenet {
namespace runtime {

// KV 分配器统一接口。
//
// 功能：
//   为每个请求维护逻辑块到物理块的映射，按 token 增长追加分配，并提供
//   块表/slot 查询供 Scheduler 组装 ExecutionPlan。
//
// 边界：
//   - 所有方法由单一推理线程调用，不要求内部锁；
//   - Reserve 失败时不得修改块表或空闲池；
//   - Release 必须幂等，未分配过的请求调用应返回成功或忽略。
class KvAllocator {
 public:
  virtual ~KvAllocator() = default;

  // 为请求追加若干 token 的 KV 容量。
  //
  // 参数：
  //   request_id：请求 ID；
  //   new_tokens：本次要追加的 token 数，必须大于 0。
  // 返回：
  //   成功返回 kOk；容量不足返回 kDeviceOom；参数非法返回
  //   kInvalidArgument。失败时不得改变任何分配状态。
  virtual Status Reserve(uint64_t request_id, int64_t new_tokens) = 0;

  // 释放请求占用的全部 KV 块。
  //
  // 参数：
  //   request_id：请求 ID。
  // 说明：
  //   幂等；未分配过的请求视为成功。释放顺序由实现保证（逆序归还）。
  virtual void Release(uint64_t request_id) = 0;

  // 返回当前空闲块数。
  virtual int64_t FreeBlocks() const = 0;

  // 返回每个块的 token 数；由后端 Attention tile 决定。
  virtual int32_t BlockSize() const = 0;

  // 返回 KV 能容纳的最大 token 总数（blocks × block_size）。
  virtual int64_t MaxTotalTokens() const = 0;

  // 读取请求的逻辑块表（逻辑块下标 → 物理块 ID）。
  //
  // 参数：
  //   request_id：请求 ID；
  //   table：输出参数，成功时写入块表快照。
  // 返回：
  //   成功返回 kOk；请求不存在返回 kRequestNotFound。
  virtual Status GetBlockTable(uint64_t request_id,
                              std::vector<int32_t>* table) const = 0;

  // 计算一段连续 token 的 KV 写入 slot。
  //
  // 参数：
  //   request_id：请求 ID；
  //   first_token：起始 token 下标（从 0 开始）；
  //   count：token 数；
  //   slots：输出参数，长度为 count。
  // 返回：
  //   成功返回 kOk；请求不存在或区间越界返回 kInvalidArgument。
  virtual Status GetSlots(uint64_t request_id, int64_t first_token,
                          int64_t count, std::vector<int64_t>* slots) const = 0;
};

}  // namespace runtime
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_COMMON_KV_ALLOCATOR_H_
