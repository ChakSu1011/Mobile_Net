// 模块：Session OpenAI 协议适配实现
//
// 职责：
//   实现 Chat Completions 请求解析与响应序列化。
//
// 边界：
//   不处理 HTTP/Engine；不实现 tools/多模态/logprobs。

#include "Service/session/openai_api.h"

#include <cstdio>
#include <ctime>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "Service/session/json_writer.h"

namespace mobilenet {
namespace service {
namespace {

// 设置 API 错误。
void SetError(ApiError* error, I32 status, const std::string& type,
              const std::string& code, const std::string& message) {
  error->status = status;
  error->type = type;
  error->code = code;
  error->message = message;
}

// 参数错误。
Status InvalidParameter(ApiError* error, const std::string& message,
                        const std::string& code = "invalid_parameter") {
  SetError(error, 400, "invalid_request_error", code, message);
  return MakeStatus(ErrorCode::kInvalidArgument, message);
}

// 不支持参数。
Status UnsupportedParameter(ApiError* error, const std::string& name) {
  SetError(error, 400, "invalid_request_error", "unsupported_parameter",
           "unsupported parameter: " + name);
  return MakeStatus(ErrorCode::kInvalidArgument,
                    "unsupported parameter: " + name);
}

// 读取可选布尔。
Bool ReadBool(const runtime::JsonValue& object, const std::string& key,
              Bool default_value, Bool* valid) {
  const runtime::JsonValue* value = object.Find(key);
  if (value == nullptr) {
    return default_value;
  }
  if (!value->IsBool()) {
    if (valid != nullptr) {
      *valid = false;
    }
    return default_value;
  }
  return value->AsBool();
}

// 校验 [minimum, maximum] 范围。
Bool InRange(FP64 value, FP64 minimum, FP64 maximum) {
  return value >= minimum && value <= maximum;
}

// 解析 messages。
Status ParseMessages(const runtime::JsonValue& messages,
                     ChatRequest* request, ApiError* error) {
  if (!messages.IsArray() || messages.AsArray().empty()) {
    return InvalidParameter(error, "messages must be a non-empty array");
  }
  for (const runtime::JsonValue& item : messages.AsArray()) {
    if (!item.IsObject()) {
      return InvalidParameter(error, "messages entries must be objects");
    }
    const runtime::JsonValue* role = item.Find("role");
    const runtime::JsonValue* content = item.Find("content");
    if (role == nullptr || !role->IsString()) {
      return InvalidParameter(error, "message.role must be a string");
    }
    if (content == nullptr) {
      return InvalidParameter(error, "message.content is required");
    }
    if (!content->IsString()) {
      return UnsupportedParameter(error, "messages.content(non-string)");
    }
    const std::string role_text = role->AsString();
    if (role_text != "system" && role_text != "user" &&
        role_text != "assistant") {
      return UnsupportedParameter(error, "messages.role=" + role_text);
    }
    request->messages.push_back({role_text, content->AsString()});
  }
  return Status::Ok();
}

// 解析 stop。
Status ParseStop(const runtime::JsonValue& body, ChatRequest* request,
                 ApiError* error) {
  const runtime::JsonValue* stop = body.Find("stop");
  if (stop == nullptr || stop->IsNull()) {
    return Status::Ok();
  }
  if (stop->IsString()) {
    if (stop->AsString().empty()) {
      return InvalidParameter(error, "stop must not be empty");
    }
    request->stop.push_back(stop->AsString());
  } else if (stop->IsArray()) {
    for (const runtime::JsonValue& item : stop->AsArray()) {
      if (!item.IsString() || item.AsString().empty()) {
        return InvalidParameter(error, "stop must contain non-empty strings");
      }
      request->stop.push_back(item.AsString());
    }
  } else {
    return InvalidParameter(error, "stop must be a string or array");
  }
  if (request->stop.size() > 4) {
    return InvalidParameter(error, "at most 4 stop strings are supported");
  }
  for (const std::string& text : request->stop) {
    if (text.size() > 256) {
      return InvalidParameter(error, "stop string is too long");
    }
  }
  return Status::Ok();
}

// 解析 stop_token_ids 扩展。
Status ParseStopTokenIds(const runtime::JsonValue& body, ChatRequest* request,
                         ApiError* error) {
  const runtime::JsonValue* ids = body.Find("stop_token_ids");
  if (ids == nullptr || ids->IsNull()) {
    return Status::Ok();
  }
  if (!ids->IsArray()) {
    return InvalidParameter(error, "stop_token_ids must be an array");
  }
  for (const runtime::JsonValue& item : ids->AsArray()) {
    if (!item.IsInteger() || item.AsInt64() < 0) {
      return InvalidParameter(error, "stop_token_ids must be non-negative ints");
    }
    request->stop_token_ids.push_back(item.AsInt64());
  }
  return Status::Ok();
}

// 解析思考模式。
Status ParseThinking(const runtime::JsonValue& body,
                     const ServerConfig& config, ChatRequest* request,
                     ApiError* error) {
  request->thinking = config.thinking;
  const runtime::JsonValue* kwargs = body.Find("chat_template_kwargs");
  if (kwargs == nullptr || kwargs->IsNull()) {
    return Status::Ok();
  }
  if (!kwargs->IsObject()) {
    return InvalidParameter(error, "chat_template_kwargs must be an object");
  }
  Bool valid = true;
  const Bool enabled =
      ReadBool(*kwargs, "enable_thinking", false, &valid);
  if (!valid) {
    return InvalidParameter(error,
                            "chat_template_kwargs.enable_thinking must be bool");
  }
  if (kwargs->Find("enable_thinking") != nullptr) {
    request->thinking =
        enabled ? ThinkingMode::kEnabled : ThinkingMode::kDisabled;
  }
  return Status::Ok();
}

}  // namespace

Status ParseChatCompletionRequest(const runtime::JsonValue& body,
                                  const ServerConfig& config,
                                  ChatRequest* request, ApiError* error) {
  if (request == nullptr || error == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "ParseChatCompletionRequest: output must not be null");
  }
  if (!body.IsObject()) {
    return InvalidParameter(error, "request body must be a JSON object");
  }
  *request = ChatRequest{};

  const runtime::JsonValue* model = body.Find("model");
  if (model != nullptr) {
    if (!model->IsString() ||
        model->AsString() != config.model_name) {
      SetError(error, 404, "invalid_request_error", "model_not_found",
               "model not found: " +
                   (model->IsString() ? model->AsString() : std::string()));
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "model not found");
    }
  }

  const runtime::JsonValue* messages = body.Find("messages");
  if (messages == nullptr) {
    return InvalidParameter(error, "messages is required");
  }
  Status status = ParseMessages(*messages, request, error);
  if (!status.ok()) {
    return status;
  }

  Bool valid = true;
  request->stream = ReadBool(body, "stream", false, &valid);
  if (!valid) {
    return InvalidParameter(error, "stream must be a boolean");
  }
  const runtime::JsonValue* stream_options = body.Find("stream_options");
  if (stream_options != nullptr && !stream_options->IsNull()) {
    if (!stream_options->IsObject()) {
      return InvalidParameter(error, "stream_options must be an object");
    }
    request->include_usage =
        ReadBool(*stream_options, "include_usage", false, &valid);
    if (!valid) {
      return InvalidParameter(
          error, "stream_options.include_usage must be a boolean");
    }
  }

  // max_completion_tokens 非 null 时优先，否则回退到 max_tokens。
  const runtime::JsonValue* max_completion =
      body.Find("max_completion_tokens");
  const runtime::JsonValue* max_tokens = body.Find("max_tokens");
  const runtime::JsonValue* chosen = nullptr;
  if (max_completion != nullptr && !max_completion->IsNull()) {
    chosen = max_completion;
  } else if (max_tokens != nullptr && !max_tokens->IsNull()) {
    chosen = max_tokens;
  }
  if (chosen != nullptr && !chosen->IsNull()) {
    if (!chosen->IsInteger() || chosen->AsInt64() <= 0 ||
        chosen->AsInt64() > (1 << 20)) {
      return InvalidParameter(error,
                              "max_tokens must be a positive integer");
    }
    request->max_tokens = static_cast<I32>(chosen->AsInt64());
  }

  const runtime::JsonValue* n = body.Find("n");
  if (n != nullptr && !n->IsNull()) {
    if (!n->IsInteger() || n->AsInt64() != 1) {
      return UnsupportedParameter(error, "n (only n=1 is supported)");
    }
  }

  // 采样参数：校验范围，但当前 Engine 为 argmax，不生效。
  request->temperature = 1.0;
  request->top_p = 1.0;
  const runtime::JsonValue* temperature = body.Find("temperature");
  if (temperature != nullptr && !temperature->IsNull()) {
    if (!temperature->IsNumber() ||
        !InRange(temperature->AsFloat64(), 0.0, 2.0)) {
      return InvalidParameter(error, "temperature must be in [0, 2]");
    }
    request->temperature = temperature->AsFloat64();
  }
  const runtime::JsonValue* top_p = body.Find("top_p");
  if (top_p != nullptr && !top_p->IsNull()) {
    if (!top_p->IsNumber() ||
        !InRange(top_p->AsFloat64(), 0.0, 1.0)) {
      return InvalidParameter(error, "top_p must be in [0, 1]");
    }
    request->top_p = top_p->AsFloat64();
  }
  const runtime::JsonValue* top_k = body.Find("top_k");
  if (top_k != nullptr && !top_k->IsNull()) {
    if (!top_k->IsInteger() || top_k->AsInt64() < 0) {
      return InvalidParameter(error, "top_k must be a non-negative integer");
    }
    request->top_k = static_cast<I32>(top_k->AsInt64());
  }
  const runtime::JsonValue* seed = body.Find("seed");
  if (seed != nullptr && !seed->IsNull()) {
    if (!seed->IsInteger()) {
      return InvalidParameter(error, "seed must be an integer");
    }
    request->seed = seed->AsInt64();
  }
  const runtime::JsonValue* frequency = body.Find("frequency_penalty");
  if (frequency != nullptr && !frequency->IsNull()) {
    if (!frequency->IsNumber() ||
        !InRange(frequency->AsFloat64(), -2.0, 2.0)) {
      return InvalidParameter(error,
                              "frequency_penalty must be in [-2, 2]");
    }
    request->frequency_penalty = frequency->AsFloat64();
  }
  const runtime::JsonValue* presence = body.Find("presence_penalty");
  if (presence != nullptr && !presence->IsNull()) {
    if (!presence->IsNumber() ||
        !InRange(presence->AsFloat64(), -2.0, 2.0)) {
      return InvalidParameter(error,
                              "presence_penalty must be in [-2, 2]");
    }
    request->presence_penalty = presence->AsFloat64();
  }

  status = ParseStop(body, request, error);
  if (!status.ok()) {
    return status;
  }
  status = ParseStopTokenIds(body, request, error);
  if (!status.ok()) {
    return status;
  }
  status = ParseThinking(body, config, request, error);
  if (!status.ok()) {
    return status;
  }

  // 明确拒绝不支持的能力。
  static const C8* kUnsupportedFields[] = {
      "tools", "tool_choice", "response_format", "logit_bias",
      "top_logprobs"};
  for (const C8* field : kUnsupportedFields) {
    const runtime::JsonValue* value = body.Find(field);
    if (value != nullptr && !value->IsNull()) {
      return UnsupportedParameter(error, field);
    }
  }
  const runtime::JsonValue* logprobs = body.Find("logprobs");
  if (logprobs != nullptr && !logprobs->IsNull()) {
    if (!logprobs->IsBool() || logprobs->AsBool()) {
      return UnsupportedParameter(error, "logprobs");
    }
  }
  return Status::Ok();
}

Status ParseSessionChatRequest(const runtime::JsonValue& body,
                               const ServerConfig& config,
                               ChatRequest* request, ApiError* error) {
  if (request == nullptr || error == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "ParseSessionChatRequest: output must not be null");
  }
  if (!body.IsObject()) {
    return InvalidParameter(error, "request body must be a JSON object");
  }
  const runtime::JsonValue* message = body.Find("message");
  if (message == nullptr || !message->IsString() ||
      message->AsString().empty()) {
    return InvalidParameter(error, "message must be a non-empty string");
  }

  // 构造等价 Chat Completions 请求，复用统一校验逻辑。
  std::vector<runtime::JsonMember> members;
  runtime::JsonValue user_message = runtime::JsonValue::MakeObject(
      {{"role", runtime::JsonValue::MakeString("user")},
       {"content", runtime::JsonValue::MakeString(message->AsString())}});
  members.push_back(
      {"messages", runtime::JsonValue::MakeArray({std::move(user_message)})});
  static const C8* kCopiedFields[] = {
      "model", "stream", "max_tokens", "max_completion_tokens",
      "temperature", "top_p", "top_k", "seed", "frequency_penalty",
      "presence_penalty", "stop", "stop_token_ids", "chat_template_kwargs",
      "stream_options"};
  for (const C8* field : kCopiedFields) {
    const runtime::JsonValue* value = body.Find(field);
    if (value != nullptr) {
      members.push_back({field, *value});
    }
  }
  runtime::JsonValue synthetic =
      runtime::JsonValue::MakeObject(std::move(members));
  return ParseChatCompletionRequest(synthetic, config, request, error);
}

std::string NewCompletionId() {
  static std::mutex mutex;
  static std::mt19937_64 generator(std::random_device{}());
  U64 value = 0;
  {
    std::lock_guard<std::mutex> lock(mutex);
    value = generator();
  }
  C8 buffer[40] = {0};
  std::snprintf(buffer, sizeof(buffer), "chatcmpl-%016llx",
                static_cast<unsigned long long>(value));
  return std::string(buffer);
}

I64 NowEpochSeconds() {
  return static_cast<I64>(std::time(nullptr));
}

std::string BuildErrorResponse(const ApiError& error) {
  JsonWriter writer;
  writer.BeginObject();
  writer.Key("error");
  writer.BeginObject();
  writer.Key("message");
  writer.String(error.message);
  writer.Key("type");
  writer.String(error.type);
  writer.Key("param");
  writer.Null();
  writer.Key("code");
  writer.String(error.code);
  writer.EndObject();
  writer.EndObject();
  return writer.str();
}

std::string BuildChatCompletionResponse(const std::string& id, I64 created,
                                        const std::string& model,
                                        const std::string& session_id,
                                        const ChatOutcome& outcome) {
  JsonWriter writer;
  writer.BeginObject();
  writer.Key("id");
  writer.String(id);
  writer.Key("object");
  writer.String("chat.completion");
  writer.Key("created");
  writer.Int(created);
  writer.Key("model");
  writer.String(model);
  if (!session_id.empty()) {
    writer.Key("session_id");
    writer.String(session_id);
  }
  writer.Key("choices");
  writer.BeginArray();
  writer.BeginObject();
  writer.Key("index");
  writer.Int(0);
  writer.Key("message");
  writer.BeginObject();
  writer.Key("role");
  writer.String("assistant");
  writer.Key("content");
  writer.String(outcome.text);
  writer.EndObject();
  writer.Key("finish_reason");
  writer.String(outcome.finish_reason);
  writer.EndObject();
  writer.EndArray();
  writer.Key("usage");
  writer.BeginObject();
  writer.Key("prompt_tokens");
  writer.Int(outcome.prompt_tokens);
  writer.Key("completion_tokens");
  writer.Int(outcome.completion_tokens);
  writer.Key("total_tokens");
  writer.Int(outcome.prompt_tokens + outcome.completion_tokens);
  writer.EndObject();
  writer.EndObject();
  return writer.str();
}

std::string BuildChatCompletionChunk(
    const std::string& id, I64 created, const std::string& model,
    const std::string& session_id, const std::string& role,
    const std::string& content, const std::string& finish_reason,
    Bool include_usage, I32 prompt_tokens, I32 completion_tokens) {
  JsonWriter writer;
  writer.BeginObject();
  writer.Key("id");
  writer.String(id);
  writer.Key("object");
  writer.String("chat.completion.chunk");
  writer.Key("created");
  writer.Int(created);
  writer.Key("model");
  writer.String(model);
  if (!session_id.empty()) {
    writer.Key("session_id");
    writer.String(session_id);
  }
  writer.Key("choices");
  writer.BeginArray();
  writer.BeginObject();
  writer.Key("index");
  writer.Int(0);
  writer.Key("delta");
  writer.BeginObject();
  if (!role.empty()) {
    writer.Key("role");
    writer.String(role);
  }
  if (!content.empty()) {
    writer.Key("content");
    writer.String(content);
  }
  writer.EndObject();
  writer.Key("finish_reason");
  if (finish_reason.empty()) {
    writer.Null();
  } else {
    writer.String(finish_reason);
  }
  writer.EndObject();
  writer.EndArray();
  if (include_usage && !finish_reason.empty()) {
    writer.Key("usage");
    writer.BeginObject();
    writer.Key("prompt_tokens");
    writer.Int(prompt_tokens);
    writer.Key("completion_tokens");
    writer.Int(completion_tokens);
    writer.Key("total_tokens");
    writer.Int(prompt_tokens + completion_tokens);
    writer.EndObject();
  }
  writer.EndObject();
  return writer.str();
}

std::string BuildModelsResponse(const std::string& model, I64 created) {
  JsonWriter writer;
  writer.BeginObject();
  writer.Key("object");
  writer.String("list");
  writer.Key("data");
  writer.BeginArray();
  writer.BeginObject();
  writer.Key("id");
  writer.String(model);
  writer.Key("object");
  writer.String("model");
  writer.Key("created");
  writer.Int(created);
  writer.Key("owned_by");
  writer.String("mobilenet");
  writer.EndObject();
  writer.EndArray();
  writer.EndObject();
  return writer.str();
}

std::string BuildSessionCreateResponse(const SessionInfo& info) {
  JsonWriter writer;
  writer.BeginObject();
  writer.Key("object");
  writer.String("session");
  writer.Key("session_id");
  writer.String(info.id);
  writer.Key("created_at");
  writer.Int(info.created_at);
  writer.Key("last_active");
  writer.Int(info.last_active);
  writer.Key("message_count");
  writer.Int(static_cast<I64>(info.message_count));
  writer.Key("state");
  writer.String(info.state);
  writer.EndObject();
  return writer.str();
}

std::string BuildSessionInfoResponse(const SessionInfo& info) {
  return BuildSessionCreateResponse(info);
}

std::string BuildSessionDeleteResponse(const std::string& session_id) {
  JsonWriter writer;
  writer.BeginObject();
  writer.Key("session_id");
  writer.String(session_id);
  writer.Key("deleted");
  writer.Boolean(true);
  writer.EndObject();
  return writer.str();
}

}  // namespace service
}  // namespace mobilenet
