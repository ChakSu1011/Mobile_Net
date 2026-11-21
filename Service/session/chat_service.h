// 模块：Session 对话编排
//
// 职责：
//   把 Chat 消息转换为 prompt token ids，提交 Engine，消费逐步响应，
//   完成流式/非流式解码、stop 截断、超时与取消。
//
// 边界：
//   - 不管理会话历史（SessionManager 负责）；
//   - 不解析 HTTP/OpenAI JSON（openai_api 负责）；
//   - 不写网络；通过 StreamSink 回调输出增量文本。
//
// 依赖：
//   Service/tokenizer、IEngine、PendingRegistry、ServerConfig。

#ifndef MOBILENET_SERVICE_SESSION_CHAT_SERVICE_H_
#define MOBILENET_SERVICE_SESSION_CHAT_SERVICE_H_

#include <functional>
#include <string>
#include <vector>

#include "Runtime/common/status.h"
#include "Service/session/engine_client.h"
#include "Service/session/pending_registry.h"
#include "Service/session/server_config.h"
#include "Service/tokenizer/tokenizer.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace service {

// 上下文超限错误前缀；HTTP 层据此映射 error code。
extern const C8 kContextLengthExceeded[];

// 对话请求（与 OpenAI 协议解耦）。
struct ChatRequest {
  // 消息列表。
  std::vector<ChatMessage> messages;

  // 是否流式；由 HTTP 层设置。
  Bool stream = false;

  // 是否在流式终 chunk 返回 usage。
  Bool include_usage = false;

  // 最大生成 token 数；0 表示使用服务默认值。
  I32 max_tokens = 0;

  // stop 字符串。
  std::vector<std::string> stop;

  // stop token id。
  std::vector<I64> stop_token_ids;

  // 思考模式。
  ThinkingMode thinking = ThinkingMode::kDisabled;

  // 采样参数仅用于日志；当前 Engine 为 argmax。
  FP64 temperature = 0.0;
  FP64 top_p = 1.0;
  I32 top_k = 0;
  FP64 frequency_penalty = 0.0;
  FP64 presence_penalty = 0.0;
  I64 seed = 0;
};

// 非流式结果。
struct ChatOutcome {
  // 生成文本。
  std::string text;

  // stop / length。
  std::string finish_reason;

  // prompt token 数。
  I32 prompt_tokens = 0;

  // 生成 token 数（含终止 token）。
  I32 completion_tokens = 0;
};

// 流式输出回调。
//
// 说明：
//   OnDelta 返回 false 表示客户端断开，ChatService 会取消 Engine 请求。
class StreamSink {
 public:
  virtual ~StreamSink() = default;

  // 输出增量文本。
  virtual Bool OnDelta(const std::string& delta) = 0;

  // 输出终态。
  virtual Bool OnFinish(const std::string& finish_reason,
                        I32 prompt_tokens, I32 completion_tokens) = 0;
};

// stop 字符串过滤器。
//
// 功能：
//   在解码文本上检测 stop，并保留必要尾部以避免跨 token 漏检。
class StopTextFilter {
 public:
  // 构造过滤器。
  //
  // 参数：
  //   stops：stop 字符串列表；空表示不过滤。
  explicit StopTextFilter(const std::vector<std::string>& stops);

  // 是否启用过滤。
  Bool enabled() const { return !stops_.empty(); }

  // 输入增量文本并输出可发送文本。
  //
  // 参数：
  //   delta：新解码文本；
  //   emit：发送函数；返回 false 表示客户端断开；
  //   hit：输出参数，true 表示命中 stop。
  // 返回：
  //   成功返回 kOk；发送失败返回 kCancelled。
  Status Feed(const std::string& delta,
              const std::function<Bool(const std::string&)>& emit,
              Bool* hit);

  // 输出剩余文本。
  Status Flush(const std::function<Bool(const std::string&)>& emit);

 private:
  // stop 列表。
  std::vector<std::string> stops_;

  // 最长 stop 字节数。
  Usize max_stop_bytes_ = 0;

  // 尚未发送的尾部文本。
  std::string pending_;
};

// 对话服务。
class ChatService {
 public:
  // 构造。
  //
  // 参数：
  //   tokenizer：已加载 Tokenizer；
  //   engine：已加载并启动的 Engine；
  //   config：服务配置；
  //   registry：请求事件注册表。
  ChatService(const Tokenizer& tokenizer, IEngine& engine,
              const ServerConfig& config, PendingRegistry* registry);

  ChatService(const ChatService&) = delete;
  ChatService& operator=(const ChatService&) = delete;

  // 非流式对话。
  Status Complete(const ChatRequest& request, ChatOutcome* outcome);

  // 流式对话。
  //
  // 参数：
  //   request：请求；
  //   sink：增量输出回调；
  //   outcome：可选；非空时同时写入完整文本与 usage，供会话历史保存。
  Status Stream(const ChatRequest& request, StreamSink* sink,
                ChatOutcome* outcome = nullptr);

  // 返回最大 prompt+生成 token 预算。
  I32 max_num_tokens() const { return config_.max_num_tokens; }

 private:
  // 构造 prompt 并按预算截断消息。
  Status Prepare(const ChatRequest& request,
                 std::vector<I64>* prompt_ids, I32* max_tokens);

  // 删除一条旧消息；成功返回 true。
  static Bool DropOldestMessage(std::vector<ChatMessage>* messages);

  // 执行一次请求。
  Status Run(const ChatRequest& request,
             const std::vector<I64>& prompt_ids, I32 max_tokens,
             StreamSink* sink, ChatOutcome* outcome);

  // 判断 token 是否为终止 token。
  Bool IsStopToken(I64 token_id, const ChatRequest& request) const;

  // Tokenizer。
  const Tokenizer& tokenizer_;

  // Engine。
  IEngine& engine_;

  // 服务配置。
  const ServerConfig& config_;

  // 事件注册表；非拥有。
  PendingRegistry* registry_ = nullptr;
};

}  // namespace service
}  // namespace mobilenet

#endif  // MOBILENET_SERVICE_SESSION_CHAT_SERVICE_H_
