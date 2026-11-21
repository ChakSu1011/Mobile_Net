// 模块：Session 会话管理
//
// 职责：
//   维护有状态多轮会话：创建/查询/删除、消息历史、隔离、忙状态与空闲
//   清理；把会话历史与新用户消息交给 ChatService。
//
// 边界：
//   - 不解析 HTTP/OpenAI JSON；
//   - 不直接调用 Engine；
//   - 同一会话串行对话，不同会话共享 ChatService。
//
// 依赖：
//   chat_service.h、server_config.h。

#ifndef MOBILENET_SERVICE_SESSION_SESSION_MANAGER_H_
#define MOBILENET_SERVICE_SESSION_SESSION_MANAGER_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "Runtime/common/status.h"
#include "Service/session/chat_service.h"
#include "Service/session/server_config.h"
#include "Service/tokenizer/tokenizer.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace service {

// 创建会话选项。
struct CreateSessionOptions {
  // 可选的自定义 session id；为空时自动生成。
  std::string session_id;

  // 可选的 system prompt。
  std::string system_prompt;
};

// 会话信息快照。
struct SessionInfo {
  // 会话 ID。
  std::string id;

  // 创建时间（Unix 秒）。
  I64 created_at = 0;

  // 最近活跃时间（Unix 秒）。
  I64 last_active = 0;

  // 消息条数。
  Usize message_count = 0;

  // 状态：idle / busy。
  std::string state = "idle";
};

// 会话管理器。
class SessionManager {
 public:
  // 构造。
  //
  // 参数：
  //   chat：已初始化的 ChatService；
  //   config：服务配置。
  SessionManager(ChatService* chat, const ServerConfig* config);

  ~SessionManager();

  SessionManager(const SessionManager&) = delete;
  SessionManager& operator=(const SessionManager&) = delete;

  // 创建会话。
  Status Create(const CreateSessionOptions& options, SessionInfo* info);

  // 查询会话。
  Status Get(const std::string& id, SessionInfo* info);

  // 删除会话。
  Status Delete(const std::string& id);

  // 有状态对话。
  //
  // 参数：
  //   id：会话 ID；
  //   user_message：本轮用户文本；
  //   request：采样/stop/流式参数；messages 字段被忽略；
  //   sink：非空时流式输出；
  //   outcome：非流式结果或流式完整文本（可空）。
  // 返回：
  //   成功返回 kOk；会话不存在返回 kRequestNotFound；忙返回 kStateError
  //   （消息前缀 session_busy）。
  Status Chat(const std::string& id, const std::string& user_message,
              const ChatRequest& request, StreamSink* sink,
              ChatOutcome* outcome);

  // 停止服务：拒绝新对话并清空会话。
  void Shutdown();

  // 返回当前会话数。
  Usize size() const;

 private:
  // 会话内部状态。
  struct Session {
    // 会话 ID。
    std::string id;

    // 消息历史；仅在持有 mutex 时读写。
    std::vector<ChatMessage> messages;

    // 创建时间（Unix 秒）。
    I64 created_epoch = 0;

    // 最近活跃时间（Unix 秒）。
    std::atomic<I64> last_active_epoch{0};

    // 消息条数快照（原子，供无锁查询）。
    std::atomic<Usize> message_count{0};

    // 是否正在对话（原子，供查询与忙检查）。
    std::atomic<Bool> busy{false};

    // 会话互斥锁；保护 messages 与串行对话。
    std::mutex mutex;
  };

  // 生成唯一 ID；调用方持有 mutex_。
  std::string GenerateIdLocked();

  // 清理空闲会话；调用方持有 mutex_。
  void PruneLocked();

  // 按 max_history_messages 裁剪会话历史；调用方持有会话锁。
  void TrimHistory(Session* session) const;

  // 填充信息快照。
  static void FillInfo(const Session& session, SessionInfo* info);

  // ChatService；非拥有。
  ChatService* chat_ = nullptr;

  // 服务配置；非拥有。
  const ServerConfig* config_ = nullptr;

  // 注册表锁。
  mutable std::mutex mutex_;

  // id → 会话。
  std::unordered_map<std::string, std::shared_ptr<Session>> sessions_;

  // 是否已停止。
  Bool shutting_down_ = false;

  // 自动 ID 计数器。
  U64 id_counter_ = 0;
};

}  // namespace service
}  // namespace mobilenet

#endif  // MOBILENET_SERVICE_SESSION_SESSION_MANAGER_H_
