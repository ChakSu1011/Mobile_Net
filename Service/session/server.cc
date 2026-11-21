// 模块：Session 服务总装实现
//
// 职责：
//   实现初始化/启动/停止、路由注册、OpenAI/会话端点、流式 SSE 与错误映射。
//
// 边界：
//   不实现推理/分词；HTTP 细节由 HttpServer 负责。

#include "Service/session/server.h"

#include <algorithm>
#include <chrono>
#include <string>
#include <utility>
#include <vector>

#include "Service/session/json_writer.h"
#include "Service/session/log.h"
#include "Service/session/openapi_spec.h"
#include "Service/session/openai_api.h"

namespace mobilenet {
namespace service {
namespace {

// 判断字符串前缀。
Bool HasPrefix(const std::string& text, const std::string& prefix) {
  return text.size() >= prefix.size() &&
         text.compare(0, prefix.size(), prefix) == 0;
}

// Engine 状态名称。
const C8* EngineStateName(runtime::EngineState state) {
  switch (state) {
    case runtime::EngineState::kCreated:
      return "CREATED";
    case runtime::EngineState::kLoaded:
      return "LOADED";
    case runtime::EngineState::kRunning:
      return "RUNNING";
    case runtime::EngineState::kDestroying:
      return "DESTROYING";
    case runtime::EngineState::kDestroyed:
      return "DESTROYED";
  }
  return "UNKNOWN";
}

// 把 Status 映射为 API 错误。
ApiError StatusToApiError(const Status& status) {
  ApiError error;
  const std::string& message = status.message();
  if (HasPrefix(message, "context_length_exceeded")) {
    error.status = 400;
    error.type = "invalid_request_error";
    error.code = "context_length_exceeded";
    error.message = message;
    return error;
  }
  if (HasPrefix(message, "session_busy")) {
    error.status = 409;
    error.type = "invalid_request_error";
    error.code = "session_busy";
    error.message = "session is busy";
    return error;
  }
  if (HasPrefix(message, "session_not_found")) {
    error.status = 404;
    error.type = "invalid_request_error";
    error.code = "session_not_found";
    error.message = "session not found";
    return error;
  }
  if (HasPrefix(message, "session_exists")) {
    error.status = 409;
    error.type = "invalid_request_error";
    error.code = "session_exists";
    error.message = "session already exists";
    return error;
  }
  if (HasPrefix(message, "session_invalid_id")) {
    error.status = 400;
    error.type = "invalid_request_error";
    error.code = "invalid_parameter";
    error.message = "invalid session id";
    return error;
  }
  switch (status.code()) {
    case ErrorCode::kTimeout:
      error.status = 504;
      error.type = "server_error";
      error.code = "request_timeout";
      error.message = "request timed out";
      return error;
    case ErrorCode::kCancelled:
      error.status = 499;
      error.type = "server_error";
      error.code = "request_cancelled";
      error.message = "request cancelled";
      return error;
    case ErrorCode::kRequestNotFound:
      error.status = 404;
      error.type = "invalid_request_error";
      error.code = "session_not_found";
      error.message = "session not found";
      return error;
    case ErrorCode::kUnsupportedOp:
      error.status = 400;
      error.type = "invalid_request_error";
      error.code = "unsupported_parameter";
      error.message = message;
      return error;
    case ErrorCode::kInvalidArgument:
      error.status = 400;
      error.type = "invalid_request_error";
      error.code = "invalid_parameter";
      error.message = message;
      return error;
    case ErrorCode::kStateError:
      error.status = 503;
      error.type = "server_error";
      error.code = "engine_unavailable";
      error.message = message;
      return error;
    case ErrorCode::kDeviceError:
    case ErrorCode::kKernelError:
    case ErrorCode::kBufferError:
    case ErrorCode::kBackendInitError:
    case ErrorCode::kDeviceOom:
      error.status = 503;
      error.type = "server_error";
      error.code = "engine_unavailable";
      error.message = message;
      return error;
    default:
      error.status = 500;
      error.type = "server_error";
      error.code = "internal_error";
      error.message = message.empty() ? "internal error" : message;
      return error;
  }
}

// 读取路径参数。
std::string PathParam(const HttpExchange* exchange, const std::string& name) {
  const auto& params = exchange->request().params;
  const auto iterator = params.find(name);
  return iterator == params.end() ? std::string() : iterator->second;
}

// SSE 输出实现。
class SseChatSink final : public StreamSink {
 public:
  SseChatSink(HttpExchange* exchange, std::string id, I64 created,
              std::string model, std::string session_id,
              Bool include_usage)
      : exchange_(exchange),
        id_(std::move(id)),
        created_(created),
        model_(std::move(model)),
        session_id_(std::move(session_id)),
        include_usage_(include_usage) {}

  Bool started() const { return started_; }

  Bool OnDelta(const std::string& delta) override {
    if (!EnsureStarted()) {
      return false;
    }
    if (delta.empty()) {
      return true;
    }
    return SendChunk("", delta, "");
  }

  Bool OnFinish(const std::string& finish_reason, I32 prompt_tokens,
                I32 completion_tokens) override {
    if (!EnsureStarted()) {
      return false;
    }
    if (!SendChunk("", "", finish_reason, prompt_tokens, completion_tokens)) {
      return false;
    }
    if (!exchange_->SendSseEvent("[DONE]")) {
      connected_ = false;
      return false;
    }
    exchange_->EndSse();
    finished_ = true;
    return true;
  }

 private:
  Bool EnsureStarted() {
    if (started_) {
      return true;
    }
    if (!exchange_->BeginSse()) {
      connected_ = false;
      return false;
    }
    started_ = true;
    return SendChunk("assistant", "", "");
  }

  Bool SendChunk(const std::string& role, const std::string& content,
                 const std::string& finish_reason,
                 I32 prompt_tokens = 0, I32 completion_tokens = 0) {
    const std::string json = BuildChatCompletionChunk(
        id_, created_, model_, session_id_, role, content, finish_reason,
        include_usage_ && !finish_reason.empty(), prompt_tokens,
        completion_tokens);
    if (!exchange_->SendSseEvent(json)) {
      connected_ = false;
      return false;
    }
    return true;
  }

  HttpExchange* exchange_ = nullptr;
  std::string id_;
  I64 created_ = 0;
  std::string model_;
  std::string session_id_;
  Bool include_usage_ = false;
  Bool started_ = false;
  Bool finished_ = false;
  Bool connected_ = true;
};

}  // namespace

Server::Server() = default;

Server::~Server() { (void)Stop(); }

Status Server::Init(const ServerConfig& config,
                    std::unique_ptr<IEngine> engine) {
  if (initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "Server::Init: already initialized");
  }
  config_ = config;
  SetLogLevel(config_.log_level);
  // 支持同一 Server 对象 Stop 后再次 Init；HTTP 尚未启动，无在途请求。
  registry_.Reset();

  TokenizerConfig tokenizer_config;
  tokenizer_config.model_dir = config_.model_dir;
  Status status = Tokenizer::Load(tokenizer_config, &tokenizer_);
  if (!status.ok()) {
    return PrefixStatus(status, "Server::Init: Tokenizer");
  }
  if (!tokenizer_->supports_chat_template()) {
    return MakeStatus(
        ErrorCode::kUnsupportedOp,
        "Server::Init: tokenizer does not provide Qwen3 chat template");
  }

  runtime::EngineConfig engine_config;
  engine_config.package_dir = config_.package_dir;
  runtime::RuntimeConfig& runtime_config = engine_config.runtime;
  runtime_config.backend = config_.backend;
  runtime_config.device_id = config_.device_id;
  runtime_config.queue_count = 1;
  runtime_config.memory_budget_bytes = 0;
  runtime_config.kv_cache_bytes = config_.kv_cache_bytes;
  runtime_config.max_num_tokens = config_.max_num_tokens;
  runtime_config.max_concurrent_requests = config_.max_concurrent_requests;
  runtime_config.verify_weights_sha256 = config_.verify_weights_sha256;
  runtime_config.threaded = true;
  runtime_config.max_concurrent_batches = 1;
  runtime_config.recompute_attention = config_.recompute_attention;

  if (engine == nullptr) {
    if (config_.package_dir.empty()) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "Server::Init: package_dir is required");
    }
    engine_ = std::make_unique<RuntimeInferenceEngine>();
  } else {
    engine_ = std::move(engine);
  }

  engine_->SetResponseCallback([this](const runtime::Response& response) {
    registry_.OnResponse(response);
  });
  status = engine_->Load(engine_config);
  if (!status.ok()) {
    return PrefixStatus(status, "Server::Init: Engine.Load");
  }
  status = engine_->Start();
  if (!status.ok()) {
    return PrefixStatus(status, "Server::Init: Engine.Start");
  }

  chat_service_ = std::make_unique<ChatService>(
      *tokenizer_, *engine_, config_, &registry_);
  sessions_ = std::make_unique<SessionManager>(chat_service_.get(), &config_);
  started_at_ = std::chrono::steady_clock::now();
  RegisterRoutes();
  initialized_ = true;
  LogInfo("server initialized: model=" + config_.model_name +
          " max_num_tokens=" + std::to_string(config_.max_num_tokens) +
          " backend=" + config_.backend);
  return Status::Ok();
}

Status Server::Start() {
  if (!initialized_) {
    return MakeStatus(ErrorCode::kStateError,
                      "Server::Start: not initialized");
  }
  Status status = http_.Start(
      config_.host, config_.port, config_.max_connections,
      config_.max_body_bytes, config_.read_timeout_ms,
      [this](HttpExchange* exchange) { return Handle(exchange); });
  if (!status.ok()) {
    return PrefixStatus(status, "Server::Start");
  }
  LogInfo("listening on http://" + config_.host + ":" +
          std::to_string(http_.port()));
  return Status::Ok();
}

Status Server::Stop() {
  // 先唤醒等待中的 ChatService，再关闭 HTTP 连接与 Engine。
  if (sessions_ != nullptr) {
    sessions_->Shutdown();
  }
  registry_.FailAll();
  (void)http_.Stop();
  sessions_.reset();
  chat_service_.reset();
  if (engine_ != nullptr) {
    (void)engine_->Destroy();
    engine_.reset();
  }
  tokenizer_.reset();
  initialized_ = false;
  return Status::Ok();
}

U16 Server::port() const { return http_.port(); }

Bool Server::running() const { return http_.running(); }

Bool Server::Handle(HttpExchange* exchange) {
  const auto start = std::chrono::steady_clock::now();
  const Bool result = router_.Handle(exchange);
  const FP64 elapsed_ms =
      std::chrono::duration<FP64, std::milli>(
          std::chrono::steady_clock::now() - start)
          .count();
  LogInfo(exchange->request().method + " " + exchange->request().path +
          " handled=" + (result ? "1" : "0") +
          " elapsed_ms=" + std::to_string(elapsed_ms));
  return result;
}

void Server::RegisterRoutes() {
  router_.Add("GET", "/health",
              [this](HttpExchange* exchange) {
                return HandleHealth(exchange);
              });
  router_.Add("GET", "/v1/models",
              [this](HttpExchange* exchange) {
                return HandleModels(exchange);
              });
  router_.Add("POST", "/v1/chat/completions",
              [this](HttpExchange* exchange) {
                return HandleChatCompletions(exchange);
              });
  router_.Add("POST", "/v1/sessions",
              [this](HttpExchange* exchange) {
                return HandleCreateSession(exchange);
              });
  router_.Add("GET", "/v1/sessions/:session_id",
              [this](HttpExchange* exchange) {
                return HandleGetSession(exchange);
              });
  router_.Add("DELETE", "/v1/sessions/:session_id",
              [this](HttpExchange* exchange) {
                return HandleDeleteSession(exchange);
              });
  router_.Add("POST", "/v1/sessions/:session_id/chat",
              [this](HttpExchange* exchange) {
                return HandleSessionChat(exchange);
              });
  router_.Add("GET", "/openapi.json",
              [this](HttpExchange* exchange) {
                return HandleOpenApi(exchange);
              });
}

Bool Server::ParseJsonBody(HttpExchange* exchange,
                           runtime::JsonValue* body, ApiError* error) {
  const std::string& text = exchange->request().body;
  Status status = runtime::ParseJson(text, body);
  if (!status.ok()) {
    error->status = 400;
    error->type = "invalid_request_error";
    error->code = "invalid_json";
    error->message = "invalid JSON: " + status.message();
    return false;
  }
  return true;
}

void Server::SendApiError(HttpExchange* exchange, const ApiError& error) {
  const ApiError effective = error.status == 0
                                 ? ApiError{500, "server_error",
                                            "internal_error", "internal error"}
                                 : error;
  exchange->SendJson(effective.status, BuildErrorResponse(effective));
}

void Server::SendStatusError(HttpExchange* exchange,
                             const Status& status) {
  if (status.code() == ErrorCode::kCancelled) {
    return;
  }
  SendApiError(exchange, StatusToApiError(status));
}

Bool Server::HandleHealth(HttpExchange* exchange) {
  const runtime::EngineStats stats = engine_ != nullptr
                                         ? engine_->stats()
                                         : runtime::EngineStats{};
  const runtime::EngineState state = engine_ != nullptr
                                         ? engine_->state()
                                         : runtime::EngineState::kCreated;
  const I64 uptime = std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::steady_clock::now() - started_at_)
                         .count();
  JsonWriter writer;
  writer.BeginObject();
  writer.Key("status");
  writer.String(state == runtime::EngineState::kRunning ? "ok" : "degraded");
  writer.Key("engine_state");
  writer.String(EngineStateName(state));
  writer.Key("model");
  writer.String(config_.model_name);
  writer.Key("uptime_s");
  writer.Int(uptime);
  writer.Key("stats");
  writer.BeginObject();
  writer.Key("total_steps");
  writer.Int(static_cast<I64>(stats.total_steps));
  writer.Key("total_requests");
  writer.Int(static_cast<I64>(stats.total_requests));
  writer.Key("total_tokens");
  writer.Int(static_cast<I64>(stats.total_tokens));
  writer.Key("running_requests");
  writer.Int(stats.running_requests);
  writer.Key("waiting_requests");
  writer.Int(stats.waiting_requests);
  writer.Key("kv_used_blocks");
  writer.Int(stats.kv_used_blocks);
  writer.Key("kv_free_blocks");
  writer.Int(stats.kv_free_blocks);
  writer.EndObject();
  writer.EndObject();
  return exchange->SendJson(200, writer.str());
}

Bool Server::HandleModels(HttpExchange* exchange) {
  return exchange->SendJson(
      200, BuildModelsResponse(config_.model_name, NowEpochSeconds()));
}

Bool Server::HandleOpenApi(HttpExchange* exchange) {
  return exchange->SendJson(200, OpenApiSpecJson());
}

Bool Server::HandleChatCompletions(HttpExchange* exchange) {
  runtime::JsonValue body;
  ApiError error;
  if (!ParseJsonBody(exchange, &body, &error)) {
    SendApiError(exchange, error);
    return true;
  }
  ChatRequest request;
  Status status =
      ParseChatCompletionRequest(body, config_, &request, &error);
  if (!status.ok()) {
    SendApiError(exchange, error);
    return true;
  }
  return RunChat(exchange, request, "", false);
}

Bool Server::HandleCreateSession(HttpExchange* exchange) {
  runtime::JsonValue body;
  ApiError error;
  const std::string& text = exchange->request().body;
  if (text.find_first_not_of(" \t\r\n") == std::string::npos) {
    body = runtime::JsonValue::MakeObject({});
  } else if (!ParseJsonBody(exchange, &body, &error)) {
    SendApiError(exchange, error);
    return true;
  }
  if (!body.IsObject()) {
    error.status = 400;
    error.code = "invalid_json";
    error.message = "request body must be an object";
    SendApiError(exchange, error);
    return true;
  }
  CreateSessionOptions options;
  const runtime::JsonValue* session_id = body.Find("session_id");
  if (session_id != nullptr) {
    if (!session_id->IsString()) {
      error.status = 400;
      error.code = "invalid_parameter";
      error.message = "session_id must be a string";
      SendApiError(exchange, error);
      return true;
    }
    options.session_id = session_id->AsString();
  }
  const runtime::JsonValue* system_prompt = body.Find("system_prompt");
  if (system_prompt != nullptr) {
    if (!system_prompt->IsString()) {
      error.status = 400;
      error.code = "invalid_parameter";
      error.message = "system_prompt must be a string";
      SendApiError(exchange, error);
      return true;
    }
    options.system_prompt = system_prompt->AsString();
  }
  SessionInfo info;
  Status status = sessions_->Create(options, &info);
  if (!status.ok()) {
    SendStatusError(exchange, status);
    return true;
  }
  return exchange->SendJson(200, BuildSessionCreateResponse(info));
}

Bool Server::HandleGetSession(HttpExchange* exchange) {
  const std::string id = PathParam(exchange, "session_id");
  SessionInfo info;
  Status status = sessions_->Get(id, &info);
  if (!status.ok()) {
    SendStatusError(exchange, status);
    return true;
  }
  return exchange->SendJson(200, BuildSessionInfoResponse(info));
}

Bool Server::HandleDeleteSession(HttpExchange* exchange) {
  const std::string id = PathParam(exchange, "session_id");
  Status status = sessions_->Delete(id);
  if (!status.ok()) {
    SendStatusError(exchange, status);
    return true;
  }
  return exchange->SendJson(200, BuildSessionDeleteResponse(id));
}

Bool Server::HandleSessionChat(HttpExchange* exchange) {
  const std::string id = PathParam(exchange, "session_id");
  runtime::JsonValue body;
  ApiError error;
  if (!ParseJsonBody(exchange, &body, &error)) {
    SendApiError(exchange, error);
    return true;
  }
  ChatRequest request;
  Status status =
      ParseSessionChatRequest(body, config_, &request, &error);
  if (!status.ok()) {
    SendApiError(exchange, error);
    return true;
  }
  return RunChat(exchange, request, id, true);
}

Bool Server::RunChat(HttpExchange* exchange, const ChatRequest& request,
                     const std::string& session_id, Bool stateful) {
  const std::string completion_id = NewCompletionId();
  const I64 created = NowEpochSeconds();
  std::string user_message;
  if (stateful && !request.messages.empty()) {
    user_message = request.messages.front().content;
  }

  if (request.stream) {
    SseChatSink sink(exchange, completion_id, created, config_.model_name,
                     session_id, request.include_usage);
    ChatOutcome outcome;
    Status status;
    if (stateful) {
      status = sessions_->Chat(session_id, user_message, request, &sink,
                               &outcome);
    } else {
      status = chat_service_->Stream(request, &sink, &outcome);
    }
    if (status.ok()) {
      return true;
    }
    if (status.code() == ErrorCode::kCancelled) {
      return true;
    }
    if (!sink.started()) {
      SendStatusError(exchange, status);
      return true;
    }
    if (exchange->client_connected()) {
      exchange->SendSseEvent(BuildErrorResponse(StatusToApiError(status)));
      // 发送终止标记，避免严格 SSE 客户端一直等待。
      exchange->SendSseEvent("[DONE]");
      exchange->EndSse();
    }
    return true;
  }

  ChatOutcome outcome;
  Status status;
  if (stateful) {
    status = sessions_->Chat(session_id, user_message, request, nullptr,
                             &outcome);
  } else {
    status = chat_service_->Complete(request, &outcome);
  }
  if (!status.ok()) {
    if (status.code() == ErrorCode::kCancelled) {
      return true;
    }
    SendStatusError(exchange, status);
    return true;
  }
  return exchange->SendJson(
      200, BuildChatCompletionResponse(completion_id, created,
                                       config_.model_name, session_id,
                                       outcome));
}

}  // namespace service
}  // namespace mobilenet
