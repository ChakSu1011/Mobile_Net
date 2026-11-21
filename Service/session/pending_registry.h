// 模块：Session 请求事件注册表
//
// 职责：
//   把 Engine 的 request_id 映射到等待中的 Chat 请求，并在
//   ResponseCallback 中无阻塞地把单步事件入队。
//
// 边界：
//   - 不解析 token、不写 HTTP；
//   - 回调路径只做短临界区入队，不等待业务线程；
//   - 队列有界，溢出时置标记，由业务线程取消请求。
//
// 依赖：
//   Runtime/common/sequence.h、Runtime/common/status.h。

#ifndef MOBILENET_SERVICE_SESSION_PENDING_REGISTRY_H_
#define MOBILENET_SERVICE_SESSION_PENDING_REGISTRY_H_

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "Runtime/common/sequence.h"
#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace service {

// 单步 Engine 事件。
struct EngineEvent {
  // 本步 token id；无 token 时为 -1。
  I64 token_id = -1;

  // 本步请求状态。
  runtime::RequestStatus status = runtime::RequestStatus::kInvalid;

  // 累计生成 token 数。
  I32 generated_count = 0;

  // prompt token 数。
  I32 prompt_count = 0;
};

// 单个等待中的请求。
struct PendingRequest {
  // 事件互斥锁。
  std::mutex mutex;

  // 事件条件变量。
  std::condition_variable cv;

  // 事件队列。
  std::deque<EngineEvent> events;

  // 是否收到终态。
  Bool terminal = false;

  // 是否发生队列溢出。
  Bool overflow = false;

  // 是否因服务停止被强制失败。
  Bool failed = false;

  // 队列上限。
  static constexpr Usize kMaxEvents = 4096;
};

// 请求事件注册表。
//
// 功能：
//   分配请求 ID、注册等待状态、接收 Engine 回调、移除状态。
//
// 边界：
//   线程安全；回调路径不得阻塞。
class PendingRegistry {
 public:
  PendingRegistry();
  ~PendingRegistry();

  PendingRegistry(const PendingRegistry&) = delete;
  PendingRegistry& operator=(const PendingRegistry&) = delete;

  // 分配唯一请求 ID。
  //
  // 返回：
  //   非 0 请求 ID；耗尽时返回 0（实际不会发生）。
  U64 AllocateId();

  // 注册等待状态。
  //
  // 参数：
  //   request_id：请求 ID，非 0；
  //   state：输出参数，非空。
  // 返回：
  //   成功返回 kOk；重复 ID 或空指针返回 kInvalidArgument。
  Status Register(U64 request_id, std::shared_ptr<PendingRequest>* state);

  // 移除等待状态。
  void Remove(U64 request_id);

  // Engine 回调入口。
  //
  // 参数：
  //   response：单步响应；只拷贝必要字段。
  void OnResponse(const runtime::Response& response);

  // 标记所有等待者为失败并唤醒。
  void FailAll();

  // 返回当前等待请求数。
  Usize size() const;

 private:
  // map 互斥锁。
  mutable std::mutex mutex_;

  // request_id → 等待状态。
  std::unordered_map<U64, std::shared_ptr<PendingRequest>> pending_;

  // 下一个请求 ID。
  U64 next_id_ = 1;

  // 是否已进入停止流程；停止后拒绝新注册，避免关闭竞态。
  std::atomic<Bool> shutting_down_{false};
};

}  // namespace service
}  // namespace mobilenet

#endif  // MOBILENET_SERVICE_SESSION_PENDING_REGISTRY_H_
