// 模块：Session OpenAI 协议适配
//
// 职责：
//   解析 OpenAI Chat Completions 请求子集，序列化 chat.completion、
//   chat.completion.chunk、模型列表、会话与错误响应。
//
// 边界：
//   - 只支持基础子集：文本 messages、stream、max_tokens、stop、
//     chat_template_kwargs.enable_thinking、stop_token_ids；
//   - tools/response_format/logprobs/n>1/多模态返回 400；
//   - 采样参数接受但不生效（Engine 当前 argmax）；
//   - 不处理 HTTP。
//
// 依赖：
//   chat_service.h、server_config.h、session_manager.h、Runtime JSON。

#ifndef MOBILENET_SERVICE_SESSION_OPENAI_API_H_
#define MOBILENET_SERVICE_SESSION_OPENAI_API_H_

#include <string>
#include <vector>

#include "Runtime/common/json.h"
#include "Runtime/common/status.h"
#include "Service/session/chat_service.h"
#include "Service/session/server_config.h"
#include "Service/session/session_manager.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace service {

// API 错误。
struct ApiError {
  // HTTP 状态码。
  I32 status = 400;

  // OpenAI error.type。
  std::string type = "invalid_request_error";

  // OpenAI error.code。
  std::string code = "invalid_request";

  // 错误消息。
  std::string message;
};

// 解析 Chat Completions 请求。
//
// 参数：
//   body：请求 JSON；
//   config：服务配置；
//   request：输出参数，非空；
//   error：输出参数，非空；失败时写入 HTTP 错误信息。
// 返回：
//   成功返回 kOk；失败返回 kInvalidArgument，并填充 error。
Status ParseChatCompletionRequest(const runtime::JsonValue& body,
                                  const ServerConfig& config,
                                  ChatRequest* request, ApiError* error);

// 解析原生会话聊天请求。
//
// 请求体：{"message": "...", "stream": ..., "max_tokens": ..., ...}
// 语义与 Chat Completions 相同，但用户消息由 message 字段提供。
Status ParseSessionChatRequest(const runtime::JsonValue& body,
                               const ServerConfig& config,
                               ChatRequest* request, ApiError* error);

// 生成 chat completion id。
std::string NewCompletionId();

// 返回当前 Unix 秒。
I64 NowEpochSeconds();

// 序列化错误响应体。
std::string BuildErrorResponse(const ApiError& error);

// 序列化非流式 chat.completion。
//
// 参数：
//   id/created/model/session_id：响应元信息；session_id 为空时不输出；
//   outcome：生成结果。
std::string BuildChatCompletionResponse(const std::string& id, I64 created,
                                        const std::string& model,
                                        const std::string& session_id,
                                        const ChatOutcome& outcome);

// 序列化一个流式 chunk。
//
// 参数：
//   id/created/model/session_id：响应元信息；
//   role：非空时写入 delta.role；
//   content：非空时写入 delta.content；
//   finish_reason：非空时写入 finish_reason，否则为 null；
//   prompt_tokens/completion_tokens：finish_reason 非空且 usage=true 时输出。
std::string BuildChatCompletionChunk(
    const std::string& id, I64 created, const std::string& model,
    const std::string& session_id, const std::string& role,
    const std::string& content, const std::string& finish_reason,
    Bool include_usage, I32 prompt_tokens, I32 completion_tokens);

// 序列化 /v1/models 响应。
std::string BuildModelsResponse(const std::string& model, I64 created);

// 序列化会话创建响应。
std::string BuildSessionCreateResponse(const SessionInfo& info);

// 序列化会话信息响应。
std::string BuildSessionInfoResponse(const SessionInfo& info);

// 序列化会话删除响应。
std::string BuildSessionDeleteResponse(const std::string& session_id);

}  // namespace service
}  // namespace mobilenet

#endif  // MOBILENET_SERVICE_SESSION_OPENAI_API_H_
