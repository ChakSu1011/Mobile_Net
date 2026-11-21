// 模块：Session 请求事件注册表实现
//
// 职责：
//   实现请求 ID 分配、状态注册、回调入队与停止清理。
//
// 边界：
//   OnResponse 不在 map 锁内等待；每个请求使用独立锁。

#include "Service/session/pending_registry.h"

#include <utility>
#include <vector>

namespace mobilenet {
namespace service {

PendingRegistry::PendingRegistry() = default;

PendingRegistry::~PendingRegistry() { FailAll(); }

U64 PendingRegistry::AllocateId() {
  std::lock_guard<std::mutex> lock(mutex_);
  U64 id = next_id_++;
  if (id == 0) {
    id = next_id_++;
  }
  return id;
}

Status PendingRegistry::Register(
    U64 request_id, std::shared_ptr<PendingRequest>* state) {
  if (state == nullptr || request_id == 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "PendingRegistry::Register: invalid request id");
  }
  if (shutting_down_.load()) {
    return MakeStatus(ErrorCode::kStateError,
                      "engine_unavailable: service stopping");
  }
  auto pending = std::make_shared<PendingRequest>();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutting_down_.load()) {
      return MakeStatus(ErrorCode::kStateError,
                        "engine_unavailable: service stopping");
    }
    if (pending_.find(request_id) != pending_.end()) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "PendingRegistry::Register: duplicate request id");
    }
    pending_.emplace(request_id, pending);
  }
  *state = std::move(pending);
  return Status::Ok();
}

void PendingRegistry::Remove(U64 request_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  pending_.erase(request_id);
}

void PendingRegistry::OnResponse(const runtime::Response& response) {
  std::shared_ptr<PendingRequest> pending;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iterator = pending_.find(response.request_id);
    if (iterator == pending_.end()) {
      // 已取消/已结束请求的迟到回调，忽略。
      return;
    }
    pending = iterator->second;
  }

  EngineEvent event;
  event.status = response.status;
  event.prompt_count = 0;
  event.generated_count = 0;
  if (!response.sequences.empty()) {
    const runtime::Sequence& sequence = response.sequences.front();
    event.token_id = sequence.current_token.token_id;
    event.generated_count = sequence.generated_count;
    event.prompt_count = sequence.prompt_count;
  }

  const Bool terminal =
      response.status == runtime::RequestStatus::kStopped ||
      response.status == runtime::RequestStatus::kIllegal;

  std::lock_guard<std::mutex> lock(pending->mutex);
  if (pending->events.size() >= PendingRequest::kMaxEvents) {
    pending->overflow = true;
  } else {
    pending->events.push_back(event);
  }
  if (terminal) {
    pending->terminal = true;
  }
  pending->cv.notify_one();
}

void PendingRegistry::FailAll() {
  shutting_down_.store(true);
  std::vector<std::shared_ptr<PendingRequest>> states;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    states.reserve(pending_.size());
    for (auto& entry : pending_) {
      states.push_back(entry.second);
    }
    pending_.clear();
  }
  for (const auto& state : states) {
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      state->failed = true;
    }
    state->cv.notify_all();
  }
}

void PendingRegistry::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  pending_.clear();
  shutting_down_.store(false);
}

Usize PendingRegistry::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return pending_.size();
}

}  // namespace service
}  // namespace mobilenet
