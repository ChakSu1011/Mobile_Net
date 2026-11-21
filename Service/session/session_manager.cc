// 模块：Session 会话管理实现
//
// 职责：
//   实现会话创建/查询/删除、忙状态、历史提交与空闲清理。
//
// 边界：
//   不解析 HTTP；不直接操作 Engine；同一会话串行对话。

#include "Service/session/session_manager.h"

#include <cstdio>
#include <ctime>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "Service/session/log.h"

namespace mobilenet {
namespace service {
namespace {

// 返回当前 Unix 秒。
I64 NowEpochSeconds() {
  return static_cast<I64>(std::time(nullptr));
}

// 判断 session id 是否只含安全字符。
Bool IsSafeSessionId(const std::string& id) {
  if (id.empty() || id.size() > 128) {
    return false;
  }
  for (const C8 ch : id) {
    const Bool alnum = (ch >= 'a' && ch <= 'z') ||
                       (ch >= 'A' && ch <= 'Z') ||
                       (ch >= '0' && ch <= '9');
    if (!alnum && ch != '-' && ch != '_') {
      return false;
    }
  }
  return true;
}

}  // namespace

SessionManager::SessionManager(ChatService* chat, const ServerConfig* config)
    : chat_(chat), config_(config) {}

SessionManager::~SessionManager() { Shutdown(); }

std::string SessionManager::GenerateIdLocked() {
  static std::mt19937_64 generator(
      static_cast<U64>(std::random_device{}()) ^
      static_cast<U64>(std::time(nullptr)));
  for (I32 attempt = 0; attempt < 32; ++attempt) {
    const U64 value = generator() ^ (++id_counter_ * 0x9E3779B97F4A7C15ull);
    C8 buffer[32] = {0};
    std::snprintf(buffer, sizeof(buffer), "%016llx",
                  static_cast<unsigned long long>(value));
    const std::string id(buffer);
    if (sessions_.find(id) == sessions_.end()) {
      return id;
    }
  }
  // 极端情况下的确定性回退。
  return "session-" + std::to_string(++id_counter_);
}

void SessionManager::PruneLocked() {
  if (config_ == nullptr || config_->session_idle_timeout_s == 0) {
    return;
  }
  const I64 now = NowEpochSeconds();
  const I64 timeout = static_cast<I64>(config_->session_idle_timeout_s);
  for (auto iterator = sessions_.begin(); iterator != sessions_.end();) {
    Session& session = *iterator->second;
    if (session.busy.load()) {
      ++iterator;
      continue;
    }
    if (now - session.last_active_epoch.load() <= timeout) {
      ++iterator;
      continue;
    }
    std::unique_lock<std::mutex> lock(session.mutex, std::try_to_lock);
    if (!lock.owns_lock() || session.busy.load()) {
      ++iterator;
      continue;
    }
    iterator = sessions_.erase(iterator);
  }
}

void SessionManager::FillInfo(const Session& session, SessionInfo* info) {
  info->id = session.id;
  info->created_at = session.created_epoch;
  info->last_active = session.last_active_epoch.load();
  info->message_count = session.message_count.load();
  info->state = session.busy.load() ? "busy" : "idle";
}

Status SessionManager::Create(const CreateSessionOptions& options,
                              SessionInfo* info) {
  if (info == nullptr || chat_ == nullptr || config_ == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "SessionManager::Create: invalid argument");
  }
  auto session = std::make_shared<Session>();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutting_down_) {
      return MakeStatus(ErrorCode::kStateError,
                        "SessionManager::Create: shutting down");
    }
    PruneLocked();
    if (sessions_.size() >= config_->max_sessions) {
      return MakeStatus(ErrorCode::kStateError,
                        "SessionManager::Create: max_sessions reached");
    }
    if (!options.session_id.empty()) {
      if (!IsSafeSessionId(options.session_id)) {
        return MakeStatus(ErrorCode::kInvalidArgument,
                          "session_invalid_id");
      }
      if (sessions_.find(options.session_id) != sessions_.end()) {
        return MakeStatus(ErrorCode::kInvalidArgument,
                          "session_exists");
      }
      session->id = options.session_id;
    } else {
      session->id = GenerateIdLocked();
    }
    session->created_epoch = NowEpochSeconds();
    session->last_active_epoch.store(session->created_epoch);
    if (!options.system_prompt.empty()) {
      session->messages.push_back({"system", options.system_prompt});
    }
    session->message_count.store(session->messages.size());
    sessions_.emplace(session->id, session);
  }
  FillInfo(*session, info);
  return Status::Ok();
}

Status SessionManager::Get(const std::string& id, SessionInfo* info) {
  if (info == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "SessionManager::Get: info must not be null");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  PruneLocked();
  const auto iterator = sessions_.find(id);
  if (iterator == sessions_.end()) {
    return MakeStatus(ErrorCode::kRequestNotFound, "session_not_found");
  }
  FillInfo(*iterator->second, info);
  return Status::Ok();
}

Status SessionManager::Delete(const std::string& id) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto iterator = sessions_.find(id);
  if (iterator == sessions_.end()) {
    return MakeStatus(ErrorCode::kRequestNotFound, "session_not_found");
  }
  Session& session = *iterator->second;
  std::unique_lock<std::mutex> session_lock(session.mutex, std::try_to_lock);
  if (!session_lock.owns_lock() || session.busy.load()) {
    return MakeStatus(ErrorCode::kStateError, "session_busy");
  }
  sessions_.erase(iterator);
  return Status::Ok();
}

Status SessionManager::Chat(const std::string& id,
                            const std::string& user_message,
                            const ChatRequest& request, StreamSink* sink,
                            ChatOutcome* outcome) {
  if (chat_ == nullptr) {
    return MakeStatus(ErrorCode::kStateError,
                      "SessionManager::Chat: chat service is null");
  }
  if (user_message.empty()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "SessionManager::Chat: message must not be empty");
  }
  std::shared_ptr<Session> session;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutting_down_) {
      return MakeStatus(ErrorCode::kStateError,
                        "SessionManager::Chat: shutting down");
    }
    PruneLocked();
    const auto iterator = sessions_.find(id);
    if (iterator == sessions_.end()) {
      return MakeStatus(ErrorCode::kRequestNotFound, "session_not_found");
    }
    session = iterator->second;
  }

  std::unique_lock<std::mutex> session_lock(session->mutex,
                                            std::try_to_lock);
  if (!session_lock.owns_lock() || session->busy.load()) {
    return MakeStatus(ErrorCode::kStateError, "session_busy");
  }
  session->busy.store(true);
  session->last_active_epoch.store(NowEpochSeconds());

  ChatRequest effective = request;
  effective.messages = session->messages;
  effective.messages.push_back({"user", user_message});

  ChatOutcome local_outcome;
  ChatOutcome* target_outcome = outcome != nullptr ? outcome : &local_outcome;
  Status status;
  if (sink != nullptr) {
    status = chat_->Stream(effective, sink, target_outcome);
  } else {
    status = chat_->Complete(effective, target_outcome);
  }

  if (status.ok()) {
    session->messages.push_back({"user", user_message});
    session->messages.push_back(
        {"assistant", target_outcome->text});
    session->message_count.store(session->messages.size());
  }
  session->last_active_epoch.store(NowEpochSeconds());
  session->busy.store(false);
  return status;
}

void SessionManager::Shutdown() {
  std::lock_guard<std::mutex> lock(mutex_);
  shutting_down_ = true;
  sessions_.clear();
}

Usize SessionManager::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return sessions_.size();
}

}  // namespace service
}  // namespace mobilenet
