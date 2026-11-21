// 模块：Session 集成测试
//
// 职责：
//   使用 FakeEngine 与真实 Qwen3 Tokenizer 验证 HTTP/OpenAI/会话/流式/
//   错误/并发/超时路径，避免加载 1.2 GB 模型。
//
// 边界：
//   - 只访问本机回环地址；
//   - 不加载真实 Engine；OCL 冒烟测试不在本文件；
//   - Tokenizer 资产缺失时返回 77（CTest 跳过）。
//
// 用法：
//   session_test <模型目录>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Runtime/common/file_util.h"
#include "Runtime/common/json.h"
#include "Runtime/common/sequence.h"
#include "Service/session/engine_client.h"
#include "Service/session/server.h"
#include "Service/session/server_config.h"
#include "Service/tokenizer/tokenizer.h"
#include "Utils/cpp/types.h"

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {

// 失败计数。
I32 g_failures = 0;

// 检查条件。
void Expect(Bool condition, const std::string& message) {
  if (!condition) {
    std::printf("FAIL: %s\n", message.c_str());
    ++g_failures;
  }
}

// 睡眠。
void SleepMs(I32 milliseconds) {
  std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

#if defined(_WIN32)
using TestSocket = SOCKET;
constexpr TestSocket kInvalidTestSocket = INVALID_SOCKET;
#else
using TestSocket = I32;
constexpr TestSocket kInvalidTestSocket = -1;
#endif

// 关闭测试 socket。
void CloseTestSocket(TestSocket socket) {
#if defined(_WIN32)
  closesocket(socket);
#else
  close(socket);
#endif
}

// 发送全部字节。
Bool SendAll(TestSocket socket, const std::string& data) {
  Usize sent = 0;
  while (sent < data.size()) {
#if defined(_WIN32)
    const I32 result = send(socket, data.data() + sent,
                            static_cast<I32>(data.size() - sent), 0);
#else
    const I32 result = static_cast<I32>(
        send(socket, data.data() + sent, data.size() - sent, 0));
#endif
    if (result <= 0) {
      return false;
    }
    sent += static_cast<Usize>(result);
  }
  return true;
}

// HTTP 响应。
struct HttpResponse {
  I32 status = 0;
  std::string body;
  std::unordered_map<std::string, std::string> headers;
};

// 解码 chunked 响应体。
Bool DecodeChunked(std::string* body) {
  std::string decoded;
  Usize position = 0;
  while (position < body->size()) {
    const Usize line_end = body->find("\r\n", position);
    if (line_end == std::string::npos) {
      return false;
    }
    std::string size_text = body->substr(position, line_end - position);
    const Usize semicolon = size_text.find(';');
    if (semicolon != std::string::npos) {
      size_text.resize(semicolon);
    }
    C8* end = nullptr;
    const unsigned long size =
        std::strtoul(size_text.c_str(), &end, 16);
    if (end == size_text.c_str()) {
      return false;
    }
    position = line_end + 2;
    if (size == 0) {
      *body = std::move(decoded);
      return true;
    }
    if (position + size > body->size()) {
      return false;
    }
    decoded.append(body->substr(position, size));
    position += size;
    if (body->compare(position, 2, "\r\n") != 0) {
      return false;
    }
    position += 2;
  }
  return false;
}

// 发送 HTTP 请求。
Bool HttpCall(U16 port, const std::string& method, const std::string& path,
              const std::string& body, HttpResponse* response,
              U32 timeout_ms = 10000) {
#if defined(_WIN32)
  static std::once_flag wsa_once;
  std::call_once(wsa_once, []() {
    WSADATA data;
    WSAStartup(MAKEWORD(2, 2), &data);
  });
#endif
  const TestSocket socket = ::socket(AF_INET, SOCK_STREAM, 0);
  if (socket == kInvalidTestSocket) {
    return false;
  }
#if defined(_WIN32)
  const DWORD timeout = static_cast<DWORD>(timeout_ms);
  setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO,
             reinterpret_cast<const C8*>(&timeout), sizeof(timeout));
#else
  struct timeval timeout;
  timeout.tv_sec = timeout_ms / 1000;
  timeout.tv_usec = (timeout_ms % 1000) * 1000;
  setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
#endif

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
  if (::connect(socket, reinterpret_cast<sockaddr*>(&address),
                sizeof(address)) != 0) {
    CloseTestSocket(socket);
    return false;
  }

  std::string request;
  request.append(method);
  request.push_back(' ');
  request.append(path);
  request.append(" HTTP/1.1\r\nHost: 127.0.0.1\r\n");
  request.append("Content-Type: application/json\r\n");
  request.append("Content-Length: ");
  request.append(std::to_string(body.size()));
  request.append("\r\nConnection: close\r\n\r\n");
  request.append(body);
  if (!SendAll(socket, request)) {
    CloseTestSocket(socket);
    return false;
  }

  std::string raw;
  C8 chunk[4096] = {0};
  while (true) {
    const I32 received = recv(socket, chunk, sizeof(chunk), 0);
    if (received <= 0) {
      break;
    }
    raw.append(chunk, static_cast<Usize>(received));
  }
  CloseTestSocket(socket);

  const Usize header_end = raw.find("\r\n\r\n");
  if (header_end == std::string::npos) {
    return false;
  }
  const std::string header_block = raw.substr(0, header_end);
  const Usize line_end = header_block.find("\r\n");
  const std::string status_line =
      header_block.substr(0, line_end == std::string::npos
                                 ? std::string::npos
                                 : line_end);
  const Usize first_space = status_line.find(' ');
  if (first_space == std::string::npos) {
    return false;
  }
  response->status = std::atoi(status_line.c_str() + first_space + 1);
  response->headers.clear();
  Usize position = line_end == std::string::npos ? header_block.size()
                                                 : line_end + 2;
  while (position < header_block.size()) {
    const Usize next = header_block.find("\r\n", position);
    const std::string line = header_block.substr(
        position, next == std::string::npos ? std::string::npos
                                            : next - position);
    const Usize colon = line.find(':');
    if (colon != std::string::npos) {
      std::string name = line.substr(0, colon);
      for (C8& ch : name) {
        if (ch >= 'A' && ch <= 'Z') {
          ch = static_cast<C8>(ch - 'A' + 'a');
        }
      }
      response->headers[name] = line.substr(colon + 2);
    }
    if (next == std::string::npos) {
      break;
    }
    position = next + 2;
  }
  response->body = raw.substr(header_end + 4);
  const auto transfer = response->headers.find("transfer-encoding");
  if (transfer != response->headers.end() &&
      transfer->second.find("chunked") != std::string::npos) {
    if (!DecodeChunked(&response->body)) {
      return false;
    }
  }
  return true;
}

// 解析 JSON。
Bool ParseJsonText(const std::string& text, mobilenet::runtime::JsonValue* value) {
  const mobilenet::Status status =
      mobilenet::runtime::ParseJson(text, value);
  if (!status.ok()) {
    std::printf("JSON ERROR: %s\n", status.message().c_str());
    return false;
  }
  return true;
}

// 读取字符串字段。
std::string GetString(const mobilenet::runtime::JsonValue& value,
                      const std::string& key) {
  const mobilenet::runtime::JsonValue* field = value.Find(key);
  return field != nullptr && field->IsString() ? field->AsString()
                                               : std::string();
}

// 读取整数。
I64 GetInt(const mobilenet::runtime::JsonValue& value,
           const std::string& key) {
  const mobilenet::runtime::JsonValue* field = value.Find(key);
  return field != nullptr && field->IsInteger() ? field->AsInt64() : -1;
}

// 从 chat.completion 中读取 content。
std::string ChoiceContent(const mobilenet::runtime::JsonValue& root) {
  const mobilenet::runtime::JsonValue* choices = root.Find("choices");
  if (choices == nullptr || !choices->IsArray() ||
      choices->AsArray().empty()) {
    return std::string();
  }
  const mobilenet::runtime::JsonValue& choice = choices->AsArray()[0];
  const mobilenet::runtime::JsonValue* message = choice.Find("message");
  return message == nullptr ? std::string() : GetString(*message, "content");
}

// 从 chat.completion 中读取 finish_reason。
std::string ChoiceFinishReason(const mobilenet::runtime::JsonValue& root) {
  const mobilenet::runtime::JsonValue* choices = root.Find("choices");
  if (choices == nullptr || !choices->IsArray() ||
      choices->AsArray().empty()) {
    return std::string();
  }
  return GetString(choices->AsArray()[0], "finish_reason");
}

// 解析 SSE 事件。
std::vector<std::string> ParseSse(const std::string& body) {
  std::vector<std::string> events;
  Usize position = 0;
  while (position < body.size()) {
    const Usize end = body.find("\n\n", position);
    const std::string event = body.substr(
        position, end == std::string::npos ? std::string::npos
                                           : end - position);
    if (event.rfind("data: ", 0) == 0) {
      events.push_back(event.substr(6));
    }
    if (end == std::string::npos) {
      break;
    }
    position = end + 2;
  }
  return events;
}

// 拼接 SSE content。
std::string SseContent(const std::vector<std::string>& events) {
  std::string content;
  for (const std::string& event : events) {
    if (event == "[DONE]") {
      continue;
    }
    mobilenet::runtime::JsonValue root;
    if (!ParseJsonText(event, &root)) {
      continue;
    }
    const mobilenet::runtime::JsonValue* choices = root.Find("choices");
    if (choices == nullptr || !choices->IsArray() ||
        choices->AsArray().empty()) {
      continue;
    }
    const mobilenet::runtime::JsonValue* delta =
        choices->AsArray()[0].Find("delta");
    if (delta != nullptr) {
      content.append(GetString(*delta, "content"));
    }
  }
  return content;
}

// 读取 SSE finish_reason。
std::string SseFinishReason(const std::vector<std::string>& events) {
  std::string reason;
  for (const std::string& event : events) {
    if (event == "[DONE]") {
      continue;
    }
    mobilenet::runtime::JsonValue root;
    if (!ParseJsonText(event, &root)) {
      continue;
    }
    const mobilenet::runtime::JsonValue* choices = root.Find("choices");
    if (choices == nullptr || !choices->IsArray() ||
        choices->AsArray().empty()) {
      continue;
    }
    const std::string current =
        GetString(choices->AsArray()[0], "finish_reason");
    if (!current.empty()) {
      reason = current;
    }
  }
  return reason;
}

// 读取 SSE 终 chunk 的 completion_tokens；无 usage 返回 -1。
I64 SseCompletionTokens(const std::vector<std::string>& events) {
  for (const std::string& event : events) {
    if (event == "[DONE]") {
      continue;
    }
    mobilenet::runtime::JsonValue root;
    if (!ParseJsonText(event, &root)) {
      continue;
    }
    const mobilenet::runtime::JsonValue* usage = root.Find("usage");
    if (usage != nullptr) {
      return GetInt(*usage, "completion_tokens");
    }
  }
  return -1;
}

// FakeEngine：按固定 token 序列逐步回调。
class FakeEngine final : public mobilenet::service::IEngine {
 public:
  struct Options {
    std::vector<I64> output_ids;
    I32 delay_ms = 0;
  };

  explicit FakeEngine(Options options) : options_(std::move(options)) {}

  ~FakeEngine() override { (void)Destroy(); }

  mobilenet::Status Load(
      const mobilenet::runtime::EngineConfig& config) override {
    (void)config;
    return mobilenet::Status::Ok();
  }

  mobilenet::Status Start() override {
    running_ = true;
    return mobilenet::Status::Ok();
  }

  mobilenet::Status AddRequest(const mobilenet::runtime::Request& request,
                               U64* request_id) override {
    if (request_id == nullptr) {
      return mobilenet::MakeStatus(mobilenet::ErrorCode::kInvalidArgument,
                                   "request_id is null");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
      return mobilenet::MakeStatus(mobilenet::ErrorCode::kStateError,
                                   "fake engine not running");
    }
    const U64 id = request.request_id != 0 ? request.request_id : next_id_++;
    *request_id = id;
    I32 count = static_cast<I32>(options_.output_ids.size());
    if (request.sample_param.max_tokens > 0) {
      count = std::min(count, request.sample_param.max_tokens);
    }
    const I32 prompt_count =
        static_cast<I32>(request.prompt_token_ids.size());
    active_.fetch_add(1);
    std::thread([this, id, count, prompt_count]() {
      I32 generated = 0;
      for (I32 index = 0; index < count; ++index) {
        if (options_.delay_ms > 0) {
          SleepMs(options_.delay_ms);
        }
        if (stopping_.load() || IsCancelled(id)) {
          active_.fetch_sub(1);
          return;
        }
        ++generated;
        const Bool terminal = index + 1 == count;
        Emit(id, options_.output_ids[static_cast<Usize>(index)],
             terminal ? mobilenet::runtime::RequestStatus::kStopped
                      : mobilenet::runtime::RequestStatus::kRunning,
             generated, prompt_count);
      }
      if (count == 0) {
        Emit(id, -1, mobilenet::runtime::RequestStatus::kStopped, 0,
             prompt_count);
      }
      active_.fetch_sub(1);
    }).detach();
    return mobilenet::Status::Ok();
  }

  mobilenet::Status CancelRequest(U64 request_id) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      cancelled_[request_id] = true;
    }
    Emit(request_id, -1, mobilenet::runtime::RequestStatus::kStopped, 0, 0);
    return mobilenet::Status::Ok();
  }

  void SetResponseCallback(
      mobilenet::runtime::ResponseCallback callback) override {
    callback_ = std::move(callback);
  }

  mobilenet::runtime::EngineStats stats() const override {
    return mobilenet::runtime::EngineStats{};
  }

  mobilenet::runtime::EngineState state() const override {
    return running_ ? mobilenet::runtime::EngineState::kRunning
                    : mobilenet::runtime::EngineState::kDestroyed;
  }

  mobilenet::Status Destroy() override {
    stopping_.store(true);
    while (active_.load() > 0) {
      SleepMs(10);
    }
    running_ = false;
    return mobilenet::Status::Ok();
  }

 private:
  Bool IsCancelled(U64 request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    return cancelled_[request_id];
  }

  void Emit(U64 request_id, I64 token_id,
            mobilenet::runtime::RequestStatus status, I32 generated,
            I32 prompt_count) {
    if (!callback_) {
      return;
    }
    mobilenet::runtime::Response response;
    response.request_id = request_id;
    response.status = status;
    mobilenet::runtime::Sequence sequence;
    sequence.current_token.token_id = token_id;
    sequence.generated_count = generated;
    sequence.prompt_count = prompt_count;
    sequence.vocab_size = 151936;
    response.sequences.push_back(sequence);
    callback_(response);
  }

  Options options_;
  std::mutex mutex_;
  std::atomic<Bool> running_{false};
  std::atomic<Bool> stopping_{false};
  std::atomic<I32> active_{0};
  U64 next_id_ = 1;
  std::unordered_map<U64, Bool> cancelled_;
  mobilenet::runtime::ResponseCallback callback_;
};

// 构造测试配置。
mobilenet::service::ServerConfig MakeConfig(const std::string& model_dir,
                                            U32 request_timeout_ms,
                                            I32 default_max_tokens) {
  mobilenet::service::ServerConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.max_connections = 16;
  config.max_body_bytes = 1u << 20;
  config.request_timeout_ms = request_timeout_ms;
  config.read_timeout_ms = 30000;
  config.model_name = "qwen3_0_6b";
  config.model_dir = model_dir;
  config.package_dir.clear();
  config.max_num_tokens = 256;
  config.max_concurrent_requests = 1;
  config.default_max_tokens = default_max_tokens;
  config.thinking = mobilenet::service::ThinkingMode::kDisabled;
  config.max_sessions = 16;
  config.session_idle_timeout_s = 0;
  config.log_level = "error";
  return config;
}

// 编码辅助。
std::vector<I64> EncodeText(const mobilenet::service::Tokenizer& tokenizer,
                            const std::string& text) {
  mobilenet::service::EncodeOptions options;
  std::vector<I64> ids;
  const mobilenet::Status status = tokenizer.Encode(text, options, &ids);
  Expect(status.ok(), "encode helper failed: " + status.message());
  return ids;
}

// 通用请求体。
std::string ChatBody(const std::string& content,
                     const std::string& extra = std::string()) {
  std::string body =
      "{\"model\":\"qwen3_0_6b\",\"messages\":[{\"role\":\"user\","
      "\"content\":\"" +
      content + "\"}]";
  if (!extra.empty()) {
    body.append(",");
    body.append(extra);
  }
  body.append("}");
  return body;
}

// 检查基础端点与非流式/流式对话。
void TestBasicChat(U16 port) {
  HttpResponse response;
  Expect(HttpCall(port, "GET", "/health", "", &response), "health call failed");
  Expect(response.status == 200, "health status != 200");
  mobilenet::runtime::JsonValue health;
  Expect(ParseJsonText(response.body, &health), "health JSON parse failed");
  Expect(GetString(health, "model") == "qwen3_0_6b", "health model mismatch");
  Expect(GetString(health, "engine_state") == "RUNNING",
         "health engine state mismatch");

  Expect(HttpCall(port, "GET", "/v1/models", "", &response),
         "models call failed");
  Expect(response.status == 200, "models status != 200");
  mobilenet::runtime::JsonValue models;
  Expect(ParseJsonText(response.body, &models), "models JSON parse failed");
  const mobilenet::runtime::JsonValue* data = models.Find("data");
  Expect(data != nullptr && data->IsArray() && !data->AsArray().empty(),
         "models data missing");

  Expect(HttpCall(port, "POST", "/v1/chat/completions",
                  ChatBody("Hi"), &response),
         "chat call failed");
  Expect(response.status == 200, "chat status != 200");
  mobilenet::runtime::JsonValue chat;
  Expect(ParseJsonText(response.body, &chat), "chat JSON parse failed");
  Expect(ChoiceContent(chat) == "Hello world", "chat content mismatch");
  Expect(ChoiceFinishReason(chat) == "stop", "chat finish_reason mismatch");
  const mobilenet::runtime::JsonValue* usage = chat.Find("usage");
  Expect(usage != nullptr && GetInt(*usage, "prompt_tokens") > 0,
         "chat usage mismatch");

  // max_tokens=1 触发 length。
  Expect(HttpCall(port, "POST", "/v1/chat/completions",
                  ChatBody("Hi", "\"max_tokens\":1"), &response),
         "max_tokens call failed");
  Expect(response.status == 200, "max_tokens status != 200");
  mobilenet::runtime::JsonValue limited;
  Expect(ParseJsonText(response.body, &limited), "max_tokens JSON failed");
  Expect(ChoiceFinishReason(limited) == "length",
         "max_tokens finish_reason should be length");

  // max_completion_tokens:null 时回退到 max_tokens。
  Expect(HttpCall(port, "POST", "/v1/chat/completions",
                  ChatBody("Hi", "\"max_completion_tokens\":null,"
                                 "\"max_tokens\":1"),
                  &response),
         "max_completion null call failed");
  mobilenet::runtime::JsonValue null_limited;
  Expect(ParseJsonText(response.body, &null_limited),
         "max_completion null JSON failed");
  Expect(ChoiceFinishReason(null_limited) == "length",
         "max_completion null should fall back to max_tokens");

  Expect(HttpCall(port, "POST", "/v1/chat/completions",
                  ChatBody("Hi",
                           "\"stream\":true,\"stream_options\":"
                           "{\"include_usage\":true}"),
                  &response),
         "stream call failed");
  Expect(response.status == 200, "stream status != 200");
  const std::vector<std::string> events = ParseSse(response.body);
  Expect(!events.empty() && events.back() == "[DONE]",
         "stream DONE missing");
  Expect(SseContent(events) == "Hello world", "stream content mismatch");
  Expect(SseFinishReason(events) == "stop",
         "stream finish_reason mismatch");
  Expect(SseCompletionTokens(events) > 0, "stream usage missing");

  // stop 字符串截断。
  Expect(HttpCall(port, "POST", "/v1/chat/completions",
                  ChatBody("Hi", "\"stop\":[\"world\"]"), &response),
         "stop call failed");
  Expect(response.status == 200, "stop status != 200");
  mobilenet::runtime::JsonValue stop_result;
  Expect(ParseJsonText(response.body, &stop_result), "stop JSON parse failed");
  Expect(ChoiceContent(stop_result) == "Hello ",
         "stop content mismatch");

  // 跨 token 边界的 stop 字符串。
  Expect(HttpCall(port, "POST", "/v1/chat/completions",
                  ChatBody("Hi", "\"stop\":[\"lo wo\"]"), &response),
         "cross-token stop call failed");
  mobilenet::runtime::JsonValue cross_stop;
  Expect(ParseJsonText(response.body, &cross_stop),
         "cross-token stop JSON failed");
  Expect(ChoiceContent(cross_stop) == "Hel",
         "cross-token stop content mismatch");
}

// 检查会话端点。
void TestSessions(U16 port) {
  HttpResponse response;
  Expect(HttpCall(port, "POST", "/v1/sessions",
                  "{\"system_prompt\":\"You are helpful.\"}", &response),
         "create session failed");
  Expect(response.status == 200, "create session status != 200");
  mobilenet::runtime::JsonValue created;
  Expect(ParseJsonText(response.body, &created), "create session JSON failed");
  const std::string session_id = GetString(created, "session_id");
  Expect(!session_id.empty(), "session id missing");

  Expect(HttpCall(port, "POST",
                  "/v1/sessions/" + session_id + "/chat",
                  "{\"message\":\"Hi\"}", &response),
         "session chat failed");
  Expect(response.status == 200, "session chat status != 200");
  mobilenet::runtime::JsonValue chat;
  Expect(ParseJsonText(response.body, &chat), "session chat JSON failed");
  Expect(ChoiceContent(chat) == "Hello world",
         "session chat content mismatch");
  Expect(GetString(chat, "session_id") == session_id,
         "session id echo mismatch");

  Expect(HttpCall(port, "GET", "/v1/sessions/" + session_id, "", &response),
         "get session failed");
  mobilenet::runtime::JsonValue info;
  Expect(ParseJsonText(response.body, &info), "session info JSON failed");
  Expect(GetInt(info, "message_count") == 3,
         "session message count after one turn mismatch");

  Expect(HttpCall(port, "POST",
                  "/v1/sessions/" + session_id + "/chat",
                  "{\"message\":\"Again\"}", &response),
         "second session chat failed");
  Expect(HttpCall(port, "GET", "/v1/sessions/" + session_id, "", &response),
         "get session after second turn failed");
  Expect(ParseJsonText(response.body, &info), "session info JSON 2 failed");
  Expect(GetInt(info, "message_count") == 5,
         "session message count after two turns mismatch");

  // 有状态流式对话。
  Expect(HttpCall(port, "POST",
                  "/v1/sessions/" + session_id + "/chat",
                  "{\"message\":\"Stream\",\"stream\":true}",
                  &response),
         "session stream chat failed");
  Expect(response.status == 200, "session stream status != 200");
  const std::vector<std::string> session_events = ParseSse(response.body);
  Expect(!session_events.empty() && session_events.back() == "[DONE]",
         "session stream DONE missing");
  Expect(SseContent(session_events) == "Hello world",
         "session stream content mismatch");
  mobilenet::runtime::JsonValue first_event;
  if (!session_events.empty() &&
      ParseJsonText(session_events.front(), &first_event)) {
    Expect(GetString(first_event, "session_id") == session_id,
           "session stream id mismatch");
  }
  Expect(HttpCall(port, "GET", "/v1/sessions/" + session_id, "", &response),
         "get session after stream failed");
  Expect(ParseJsonText(response.body, &info), "session stream info JSON failed");
  Expect(GetInt(info, "message_count") == 7,
         "session message count after stream mismatch");

  // 第二个会话隔离。
  Expect(HttpCall(port, "POST", "/v1/sessions", "{}", &response),
         "create session 2 failed");
  mobilenet::runtime::JsonValue created2;
  Expect(ParseJsonText(response.body, &created2), "create session 2 JSON failed");
  const std::string session2 = GetString(created2, "session_id");
  Expect(HttpCall(port, "GET", "/v1/sessions/" + session2, "", &response),
         "get session 2 failed");
  mobilenet::runtime::JsonValue info2;
  Expect(ParseJsonText(response.body, &info2), "session 2 info JSON failed");
  Expect(GetInt(info2, "message_count") == 0,
         "session isolation mismatch");

  Expect(HttpCall(port, "DELETE", "/v1/sessions/" + session_id, "",
                  &response),
         "delete session failed");
  Expect(response.status == 200, "delete session status != 200");
  Expect(HttpCall(port, "GET", "/v1/sessions/" + session_id, "", &response),
         "get deleted session failed");
  Expect(response.status == 404, "deleted session should be 404");
}

// 检查错误路径。
void TestErrors(U16 port) {
  HttpResponse response;

  Expect(HttpCall(port, "POST", "/v1/chat/completions", "not-json",
                  &response),
         "invalid json call failed");
  Expect(response.status == 400, "invalid json status != 400");
  mobilenet::runtime::JsonValue error;
  Expect(ParseJsonText(response.body, &error), "invalid json error parse failed");
  Expect(error.Find("error") != nullptr, "error envelope missing");

  Expect(HttpCall(port, "POST", "/v1/chat/completions",
                  "{\"model\":\"other\",\"messages\":[{\"role\":\"user\","
                  "\"content\":\"Hi\"}]}",
                  &response),
         "model mismatch call failed");
  Expect(response.status == 404, "model mismatch status != 404");

  Expect(HttpCall(port, "POST", "/v1/chat/completions",
                  ChatBody("Hi", "\"tools\":[]"), &response),
         "tools call failed");
  Expect(response.status == 400, "tools status != 400");

  Expect(HttpCall(port, "POST", "/v1/chat/completions",
                  ChatBody("Hi", "\"n\":2"), &response),
         "n=2 call failed");
  Expect(response.status == 400, "n=2 status != 400");

  Expect(HttpCall(port, "POST", "/v1/sessions/missing/chat",
                  "{\"message\":\"Hi\"}", &response),
         "missing session call failed");
  Expect(response.status == 404, "missing session status != 404");

  Expect(HttpCall(port, "GET", "/v1/chat/completions", "", &response),
         "method mismatch call failed");
  Expect(response.status == 405, "method mismatch status != 405");

  // 多条历史消息，丢弃旧消息后可成功生成。
  std::string many_messages = "{\"model\":\"qwen3_0_6b\",\"messages\":[";
  for (I32 index = 0; index < 5; ++index) {
    if (index > 0) {
      many_messages.push_back(',');
    }
    many_messages.append("{\"role\":\"user\",\"content\":\"");
    many_messages.append(100, 'a');
    many_messages.append("\"}");
  }
  many_messages.append("]}");
  Expect(HttpCall(port, "POST", "/v1/chat/completions", many_messages,
                  &response),
         "truncation call failed");
  Expect(response.status == 200, "truncation status != 200");
  mobilenet::runtime::JsonValue truncated;
  Expect(ParseJsonText(response.body, &truncated),
         "truncation JSON failed");
  Expect(ChoiceContent(truncated) == "Hello world",
         "truncation content mismatch");

  std::string long_text(2000, 'a');
  Expect(HttpCall(port, "POST", "/v1/chat/completions",
                  ChatBody(long_text), &response),
         "context call failed");
  Expect(response.status == 400, "context status != 400");
  mobilenet::runtime::JsonValue context_error;
  if (ParseJsonText(response.body, &context_error)) {
    const mobilenet::runtime::JsonValue* err = context_error.Find("error");
    Expect(err != nullptr && GetString(*err, "code") ==
                                 "context_length_exceeded",
           "context error code mismatch");
  }
}

// 检查 OpenAPI。
void TestOpenApi(U16 port) {
  HttpResponse response;
  Expect(HttpCall(port, "GET", "/openapi.json", "", &response),
         "openapi call failed");
  Expect(response.status == 200, "openapi status != 200");
  mobilenet::runtime::JsonValue spec;
  Expect(ParseJsonText(response.body, &spec), "openapi JSON parse failed");
  const mobilenet::runtime::JsonValue* paths = spec.Find("paths");
  Expect(paths != nullptr && paths->Find("/v1/chat/completions") != nullptr,
         "openapi path missing");
}

// 检查 EOS/stop token 不计入 completion_tokens。
void TestStopTokenUsage(const std::string& model_dir,
                        const std::vector<I64>& output_ids) {
  std::vector<I64> with_eos = output_ids;
  with_eos.push_back(151645);
  mobilenet::service::ServerConfig config =
      MakeConfig(model_dir, 5000, 16);
  mobilenet::service::Server server;
  auto engine = std::make_unique<FakeEngine>(
      FakeEngine::Options{with_eos, 0});
  mobilenet::Status status = server.Init(config, std::move(engine));
  Expect(status.ok(), "stop token server init failed: " + status.message());
  if (!status.ok()) {
    return;
  }
  status = server.Start();
  Expect(status.ok(), "stop token server start failed: " + status.message());
  if (!status.ok()) {
    (void)server.Stop();
    return;
  }
  const U16 port = server.port();
  HttpResponse response;
  Expect(HttpCall(port, "POST", "/v1/chat/completions", ChatBody("Hi"),
                  &response),
         "stop token chat failed");
  mobilenet::runtime::JsonValue chat;
  Expect(ParseJsonText(response.body, &chat), "stop token JSON failed");
  Expect(ChoiceContent(chat) == "Hello world",
         "stop token content mismatch");
  Expect(ChoiceFinishReason(chat) == "stop",
         "stop token finish_reason mismatch");
  const mobilenet::runtime::JsonValue* usage = chat.Find("usage");
  Expect(usage != nullptr && GetInt(*usage, "completion_tokens") == 2,
         "stop token should not be counted in usage");
  (void)server.Stop();
}

// 检查会话历史上限。
void TestHistoryCap(const std::string& model_dir,
                    const std::vector<I64>& output_ids) {
  mobilenet::service::ServerConfig config =
      MakeConfig(model_dir, 5000, 16);
  config.max_history_messages = 4;
  mobilenet::service::Server server;
  auto engine = std::make_unique<FakeEngine>(
      FakeEngine::Options{output_ids, 0});
  mobilenet::Status status = server.Init(config, std::move(engine));
  Expect(status.ok(), "history cap server init failed: " + status.message());
  if (!status.ok()) {
    return;
  }
  status = server.Start();
  Expect(status.ok(), "history cap server start failed: " + status.message());
  if (!status.ok()) {
    (void)server.Stop();
    return;
  }
  const U16 port = server.port();
  HttpResponse response;
  Expect(HttpCall(port, "POST", "/v1/sessions", "{}", &response),
         "history cap create failed");
  mobilenet::runtime::JsonValue created;
  Expect(ParseJsonText(response.body, &created), "history cap create JSON");
  const std::string session_id = GetString(created, "session_id");
  for (I32 turn = 0; turn < 3; ++turn) {
    Expect(HttpCall(port, "POST", "/v1/sessions/" + session_id + "/chat",
                    "{\"message\":\"Hi\"}", &response),
           "history cap chat failed");
    Expect(response.status == 200, "history cap chat status != 200");
  }
  Expect(HttpCall(port, "GET", "/v1/sessions/" + session_id, "", &response),
         "history cap get failed");
  mobilenet::runtime::JsonValue info;
  Expect(ParseJsonText(response.body, &info), "history cap info JSON");
  Expect(GetInt(info, "message_count") == 4,
         "history cap message_count mismatch");
  (void)server.Stop();
}

// 检查 Stop 后可重新 Init/Start。
void TestRestart(const std::string& model_dir,
                 const std::vector<I64>& output_ids) {
  mobilenet::service::ServerConfig config =
      MakeConfig(model_dir, 5000, 16);
  mobilenet::service::Server server;
  auto first_engine = std::make_unique<FakeEngine>(
      FakeEngine::Options{output_ids, 0});
  mobilenet::Status status = server.Init(config, std::move(first_engine));
  Expect(status.ok(), "restart first init failed: " + status.message());
  if (status.ok()) {
    status = server.Start();
    Expect(status.ok(), "restart first start failed: " + status.message());
    (void)server.Stop();
  }
  auto second_engine = std::make_unique<FakeEngine>(
      FakeEngine::Options{output_ids, 0});
  status = server.Init(config, std::move(second_engine));
  Expect(status.ok(), "restart second init failed: " + status.message());
  if (!status.ok()) {
    return;
  }
  status = server.Start();
  Expect(status.ok(), "restart second start failed: " + status.message());
  if (!status.ok()) {
    (void)server.Stop();
    return;
  }
  HttpResponse response;
  Expect(HttpCall(server.port(), "POST", "/v1/chat/completions",
                  ChatBody("Hi"), &response),
         "restart chat failed");
  Expect(response.status == 200, "restart chat status != 200");
  (void)server.Stop();
}

// 检查同一端口不能被两个实例同时绑定。
void TestExclusiveBind(const std::string& model_dir,
                       const std::vector<I64>& output_ids) {
  mobilenet::service::ServerConfig config =
      MakeConfig(model_dir, 5000, 16);
  mobilenet::service::Server first;
  auto first_engine = std::make_unique<FakeEngine>(
      FakeEngine::Options{output_ids, 0});
  mobilenet::Status status = first.Init(config, std::move(first_engine));
  Expect(status.ok(), "exclusive bind first init failed: " + status.message());
  if (!status.ok()) {
    return;
  }
  status = first.Start();
  Expect(status.ok(), "exclusive bind first start failed: " + status.message());
  if (!status.ok()) {
    (void)first.Stop();
    return;
  }
  const U16 port = first.port();

  mobilenet::service::ServerConfig second_config = config;
  second_config.port = port;
  mobilenet::service::Server second;
  auto second_engine = std::make_unique<FakeEngine>(
      FakeEngine::Options{output_ids, 0});
  status = second.Init(second_config, std::move(second_engine));
  Expect(status.ok(), "exclusive bind second init failed: " + status.message());
  if (status.ok()) {
    status = second.Start();
    Expect(!status.ok(), "second server should fail to bind same port");
    (void)second.Stop();
  }
  (void)first.Stop();
}

// 检查请求体上限。
void TestBodyLimit(const std::string& model_dir,
                   const std::vector<I64>& output_ids) {
  mobilenet::service::ServerConfig config =
      MakeConfig(model_dir, 5000, 16);
  config.max_body_bytes = 512;
  mobilenet::service::Server server;
  auto engine = std::make_unique<FakeEngine>(
      FakeEngine::Options{output_ids, 0});
  mobilenet::Status status = server.Init(config, std::move(engine));
  Expect(status.ok(), "body limit init failed: " + status.message());
  if (!status.ok()) {
    return;
  }
  status = server.Start();
  Expect(status.ok(), "body limit start failed: " + status.message());
  if (!status.ok()) {
    (void)server.Stop();
    return;
  }
  const U16 port = server.port();
  const std::string long_body = ChatBody(std::string(1000, 'a'));
  HttpResponse response;
  (void)HttpCall(port, "POST", "/v1/chat/completions", long_body,
                 &response);
  // 服务器可能在客户端仍在发送时关闭连接；以响应状态为准。
  Expect(response.status == 413, "body limit status != 413");
  (void)server.Stop();
}

// 检查会话忙。
void TestSessionBusy(const std::string& model_dir,
                     const std::vector<I64>& output_ids) {
  mobilenet::service::ServerConfig config =
      MakeConfig(model_dir, 5000, 16);
  mobilenet::service::Server server;
  auto engine = std::make_unique<FakeEngine>(
      FakeEngine::Options{output_ids, 300});
  mobilenet::Status status = server.Init(config, std::move(engine));
  Expect(status.ok(), "busy server init failed: " + status.message());
  if (!status.ok()) {
    return;
  }
  status = server.Start();
  Expect(status.ok(), "busy server start failed: " + status.message());
  if (!status.ok()) {
    (void)server.Stop();
    return;
  }
  const U16 port = server.port();

  HttpResponse response;
  Expect(HttpCall(port, "POST", "/v1/sessions", "{}", &response),
         "busy create session failed");
  mobilenet::runtime::JsonValue created;
  Expect(ParseJsonText(response.body, &created), "busy create JSON failed");
  const std::string session_id = GetString(created, "session_id");

  HttpResponse first;
  HttpResponse second;
  std::thread worker([&]() {
    Expect(HttpCall(port, "POST", "/v1/sessions/" + session_id + "/chat",
                    "{\"message\":\"Hi\"}", &first),
           "busy first chat call failed");
  });
  SleepMs(100);
  Expect(HttpCall(port, "POST", "/v1/sessions/" + session_id + "/chat",
                  "{\"message\":\"Again\"}", &second),
         "busy second chat call failed");
  worker.join();

  Expect(first.status == 200, "busy first status != 200");
  Expect(second.status == 409, "busy second status != 409");
  mobilenet::runtime::JsonValue busy_error;
  if (ParseJsonText(second.body, &busy_error)) {
    const mobilenet::runtime::JsonValue* error = busy_error.Find("error");
    Expect(error != nullptr && GetString(*error, "code") == "session_busy",
           "busy error code mismatch");
  }
  (void)server.Stop();
}

// 检查多连接并发聊天。
void TestConcurrentChats(const std::string& model_dir,
                         const std::vector<I64>& output_ids) {
  mobilenet::service::ServerConfig config =
      MakeConfig(model_dir, 5000, 16);
  mobilenet::service::Server server;
  auto engine = std::make_unique<FakeEngine>(
      FakeEngine::Options{output_ids, 50});
  mobilenet::Status status = server.Init(config, std::move(engine));
  Expect(status.ok(), "concurrent server init failed: " + status.message());
  if (!status.ok()) {
    return;
  }
  status = server.Start();
  Expect(status.ok(), "concurrent server start failed: " + status.message());
  if (!status.ok()) {
    (void)server.Stop();
    return;
  }
  const U16 port = server.port();

  constexpr I32 kThreads = 4;
  std::vector<HttpResponse> responses(kThreads);
  std::vector<std::thread> threads;
  for (I32 index = 0; index < kThreads; ++index) {
    threads.emplace_back([port, index, &responses]() {
      (void)HttpCall(port, "POST", "/v1/chat/completions",
                     ChatBody("Concurrent " + std::to_string(index)),
                     &responses[static_cast<Usize>(index)]);
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  for (I32 index = 0; index < kThreads; ++index) {
    const HttpResponse& response = responses[static_cast<Usize>(index)];
    Expect(response.status == 200, "concurrent chat status != 200");
    mobilenet::runtime::JsonValue chat;
    if (ParseJsonText(response.body, &chat)) {
      Expect(ChoiceContent(chat) == "Hello world",
             "concurrent chat content mismatch");
    }
  }
  (void)server.Stop();
}

// 检查客户端中途断开后服务仍可用。
void TestClientDisconnect(const std::string& model_dir,
                          const std::vector<I64>& output_ids) {
  mobilenet::service::ServerConfig config =
      MakeConfig(model_dir, 5000, 16);
  mobilenet::service::Server server;
  auto engine = std::make_unique<FakeEngine>(
      FakeEngine::Options{output_ids, 100});
  mobilenet::Status status = server.Init(config, std::move(engine));
  Expect(status.ok(), "disconnect server init failed: " + status.message());
  if (!status.ok()) {
    return;
  }
  status = server.Start();
  Expect(status.ok(), "disconnect server start failed: " + status.message());
  if (!status.ok()) {
    (void)server.Stop();
    return;
  }
  const U16 port = server.port();

  HttpResponse response;
  // 客户端 1 ms 读超时后关闭连接，模拟中途断开。
  (void)HttpCall(port, "POST", "/v1/chat/completions",
                 ChatBody("Hi", "\"stream\":true"), &response, 1);
  SleepMs(600);

  Expect(HttpCall(port, "GET", "/health", "", &response),
         "disconnect health call failed");
  Expect(response.status == 200, "server unhealthy after disconnect");
  (void)server.Stop();
}

// 检查请求超时。
void TestTimeout(const std::string& model_dir,
                 const std::vector<I64>& output_ids) {
  mobilenet::service::ServerConfig config = MakeConfig(model_dir, 50, 16);
  mobilenet::service::Server server;
  auto engine = std::make_unique<FakeEngine>(
      FakeEngine::Options{output_ids, 400});
  mobilenet::Status status = server.Init(config, std::move(engine));
  Expect(status.ok(), "timeout server init failed: " + status.message());
  if (!status.ok()) {
    return;
  }
  status = server.Start();
  Expect(status.ok(), "timeout server start failed: " + status.message());
  if (!status.ok()) {
    (void)server.Stop();
    return;
  }
  const U16 port = server.port();

  HttpResponse response;
  Expect(HttpCall(port, "POST", "/v1/chat/completions", ChatBody("Hi"),
                  &response),
         "timeout chat call failed");
  Expect(response.status == 504, "timeout status != 504");

  Expect(HttpCall(port, "GET", "/health", "", &response),
         "timeout health call failed");
  Expect(response.status == 200, "server unhealthy after timeout");
  (void)server.Stop();
}

}  // namespace

// 测试入口。
I32 main(I32 argc, C8** argv) {
  if (argc < 2) {
    std::printf("usage: session_test <model_dir>\n");
    return 1;
  }
  const std::string model_dir = argv[1];
  mobilenet::service::TokenizerConfig tokenizer_config;
  tokenizer_config.model_dir = model_dir;
  std::unique_ptr<mobilenet::service::Tokenizer> tokenizer;
  mobilenet::Status status =
      mobilenet::service::Tokenizer::Load(tokenizer_config, &tokenizer);
  if (!status.ok()) {
    std::printf("SKIP: cannot load tokenizer: %s\n", status.message().c_str());
    return 77;
  }

  const std::vector<I64> output_ids = EncodeText(*tokenizer, "Hello world");
  {
    mobilenet::service::ServerConfig config =
        MakeConfig(model_dir, 5000, 16);
    mobilenet::service::Server server;
    auto engine = std::make_unique<FakeEngine>(
        FakeEngine::Options{output_ids, 0});
    status = server.Init(config, std::move(engine));
    Expect(status.ok(), "server init failed: " + status.message());
    if (status.ok()) {
      status = server.Start();
      Expect(status.ok(), "server start failed: " + status.message());
      if (status.ok()) {
        const U16 port = server.port();
        TestBasicChat(port);
        TestSessions(port);
        TestErrors(port);
        TestOpenApi(port);
      }
      (void)server.Stop();
    }
  }

  TestStopTokenUsage(model_dir, output_ids);
  TestHistoryCap(model_dir, output_ids);
  TestRestart(model_dir, output_ids);
  TestExclusiveBind(model_dir, output_ids);
  TestBodyLimit(model_dir, output_ids);
  TestSessionBusy(model_dir, output_ids);
  TestConcurrentChats(model_dir, output_ids);
  TestClientDisconnect(model_dir, output_ids);
  TestTimeout(model_dir, output_ids);

  if (g_failures != 0) {
    std::printf("%d session checks failed\n", g_failures);
    return 1;
  }
  std::printf("all session checks passed\n");
  return 0;
}
