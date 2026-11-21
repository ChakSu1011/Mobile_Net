// 模块：Session 最小 HTTP/1.1 服务器实现
//
// 职责：
//   实现 socket 监听、请求解析、普通响应、SSE chunked 响应、连接管理与
//   优雅停止。
//
// 边界：
//   只支持单请求单连接；不支持 HTTPS/HTTP2/chunked 请求体。
//
// 主要逻辑：
//   1. accept 线程接收连接，连接线程解析请求并调用 handler；
//   2. HttpExchange 负责写普通 JSON 或 SSE 事件；
//   3. Stop 先关闭监听与在途 socket，再 join 所有线程。

#include "Service/session/http_server.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
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

#include "Service/session/json_writer.h"
#include "Service/session/log.h"

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace mobilenet {
namespace service {

#if defined(_WIN32)
using SocketHandle = SOCKET;
using SocketLength = I32;
constexpr SocketHandle kInvalidSocket = INVALID_SOCKET;
#else
using SocketHandle = I32;
using SocketLength = socklen_t;
constexpr SocketHandle kInvalidSocket = -1;
#endif

namespace {

// 单次读取缓冲大小。
constexpr Usize kReadBufferSize = 4096;

// 头部最大字节数。
constexpr Usize kMaxHeaderBytes = 16 * 1024;

// 返回 HTTP 原因短语。
const C8* ReasonPhrase(I32 status) {
  switch (status) {
    case 200:
      return "OK";
    case 400:
      return "Bad Request";
    case 404:
      return "Not Found";
    case 405:
      return "Method Not Allowed";
    case 408:
      return "Request Timeout";
    case 413:
      return "Payload Too Large";
    case 415:
      return "Unsupported Media Type";
    case 500:
      return "Internal Server Error";
    case 503:
      return "Service Unavailable";
    case 504:
      return "Gateway Timeout";
    default:
      return "Error";
  }
}

// 关闭 socket。
void CloseSocket(SocketHandle socket) {
#if defined(_WIN32)
  closesocket(socket);
#else
  close(socket);
#endif
}

// 半关闭 socket，唤醒阻塞的 recv/send。
void ShutdownSocket(SocketHandle socket) {
#if defined(_WIN32)
  shutdown(socket, SD_BOTH);
#else
  shutdown(socket, SHUT_RDWR);
#endif
}

// 发送字节。
I32 SendBytes(SocketHandle socket, const C8* data, Usize size) {
#if defined(_WIN32)
  return send(socket, data, static_cast<I32>(size), 0);
#else
#if defined(MSG_NOSIGNAL)
  return static_cast<I32>(send(socket, data, size, MSG_NOSIGNAL));
#else
  return static_cast<I32>(send(socket, data, size, 0));
#endif
#endif
}

// 接收字节。
I32 RecvBytes(SocketHandle socket, C8* data, Usize size) {
#if defined(_WIN32)
  return recv(socket, data, static_cast<I32>(size), 0);
#else
  return static_cast<I32>(recv(socket, data, size, 0));
#endif
}

// 判断最近一次 recv 是否超时。
Bool IsReceiveTimeout() {
#if defined(_WIN32)
  return WSAGetLastError() == WSAETIMEDOUT;
#else
  return errno == EAGAIN || errno == EWOULDBLOCK;
#endif
}

// 设置接收/发送超时。
void SetSocketTimeout(SocketHandle socket, U32 timeout_ms) {
#if defined(_WIN32)
  const DWORD value = static_cast<DWORD>(timeout_ms);
  setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO,
             reinterpret_cast<const C8*>(&value), sizeof(value));
  setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO,
             reinterpret_cast<const C8*>(&value), sizeof(value));
#else
  struct timeval value;
  value.tv_sec = static_cast<time_t>(timeout_ms / 1000);
  value.tv_usec = static_cast<suseconds_t>((timeout_ms % 1000) * 1000);
  setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &value, sizeof(value));
  setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, &value, sizeof(value));
#endif
}

// 十六进制字符转数值。
I32 HexValue(C8 ch) {
  if (ch >= '0' && ch <= '9') {
    return ch - '0';
  }
  if (ch >= 'a' && ch <= 'f') {
    return ch - 'a' + 10;
  }
  if (ch >= 'A' && ch <= 'F') {
    return ch - 'A' + 10;
  }
  return -1;
}

// URL 解码。
std::string UrlDecode(const std::string& text) {
  std::string decoded;
  decoded.reserve(text.size());
  for (Usize index = 0; index < text.size(); ++index) {
    if (text[index] == '+') {
      decoded.push_back(' ');
      continue;
    }
    if (text[index] == '%' && index + 2 < text.size()) {
      const I32 high = HexValue(text[index + 1]);
      const I32 low = HexValue(text[index + 2]);
      if (high >= 0 && low >= 0) {
        decoded.push_back(static_cast<C8>((high << 4) | low));
        index += 2;
        continue;
      }
    }
    decoded.push_back(text[index]);
  }
  return decoded;
}

// 去除首尾空白。
std::string Trim(const std::string& text) {
  Usize begin = 0;
  Usize end = text.size();
  while (begin < end && (text[begin] == ' ' || text[begin] == '\t')) {
    ++begin;
  }
  while (end > begin && (text[end - 1] == ' ' || text[end - 1] == '\t')) {
    --end;
  }
  return text.substr(begin, end - begin);
}

// 转小写。
std::string Lowercase(std::string text) {
  for (C8& ch : text) {
    if (ch >= 'A' && ch <= 'Z') {
      ch = static_cast<C8>(ch - 'A' + 'a');
    }
  }
  return text;
}

// 构建错误 JSON。
std::string BuildErrorBody(const std::string& message,
                           const std::string& code) {
  JsonWriter writer;
  writer.BeginObject();
  writer.Key("error");
  writer.BeginObject();
  writer.Key("message");
  writer.String(message);
  writer.Key("type");
  writer.String("invalid_request_error");
  writer.Key("param");
  writer.Null();
  writer.Key("code");
  writer.String(code);
  writer.EndObject();
  writer.EndObject();
  return writer.str();
}

// 解析查询串。
void ParseQuery(const std::string& query,
                std::unordered_map<std::string, std::string>* params) {
  if (params == nullptr) {
    return;
  }
  params->clear();
  Usize begin = 0;
  while (begin <= query.size()) {
    const Usize ampersand = query.find('&', begin);
    const std::string pair =
        query.substr(begin, ampersand == std::string::npos
                               ? std::string::npos
                               : ampersand - begin);
    if (!pair.empty()) {
      const Usize equals = pair.find('=');
      if (equals == std::string::npos) {
        params->emplace(UrlDecode(pair), std::string());
      } else {
        params->emplace(UrlDecode(pair.substr(0, equals)),
                        UrlDecode(pair.substr(equals + 1)));
      }
    }
    if (ampersand == std::string::npos) {
      break;
    }
    begin = ampersand + 1;
  }
}

// 读取结果。
enum class ReadResult {
  kOk = 0,
  kClosed = 1,
  kBadRequest = 2,
  kTooLarge = 3,
  kTimeout = 4,
};

}  // namespace

// 单个连接。
struct HttpConnection {
  // socket。
  SocketHandle socket = kInvalidSocket;

  // 是否仍连接。
  std::atomic<Bool> connected{true};

  // 写全部字节。
  Bool SendRaw(const void* data, Usize size) {
    const C8* bytes = static_cast<const C8*>(data);
    Usize sent = 0;
    while (sent < size) {
      const I32 result =
          SendBytes(socket, bytes + sent, size - sent);
      if (result <= 0) {
        connected.store(false);
        return false;
      }
      sent += static_cast<Usize>(result);
    }
    return true;
  }

  // 写字符串。
  Bool Send(const std::string& text) {
    return SendRaw(text.data(), text.size());
  }
};

namespace {

// 直接从连接发送错误响应（解析失败/限流）。
void SendDirectError(const std::shared_ptr<HttpConnection>& connection,
                     I32 status, const std::string& message,
                     const std::string& code) {
  if (connection == nullptr) {
    return;
  }
  const std::string body = BuildErrorBody(message, code);
  std::string response;
  response.reserve(body.size() + 128);
  response.append("HTTP/1.1 ");
  response.append(std::to_string(status));
  response.push_back(' ');
  response.append(ReasonPhrase(status));
  response.append("\r\nContent-Type: application/json; charset=utf-8\r\n");
  response.append("Content-Length: ");
  response.append(std::to_string(body.size()));
  response.append("\r\nConnection: close\r\n\r\n");
  response.append(body);
  connection->Send(response);
}

}  // namespace

HttpExchange::HttpExchange(std::shared_ptr<HttpConnection> connection,
                           HttpRequest request)
    : connection_(std::move(connection)), request_(std::move(request)) {}

HttpExchange::~HttpExchange() = default;

void HttpExchange::AddParam(const std::string& key,
                            const std::string& value) {
  request_.params[key] = value;
}

Bool HttpExchange::SendJson(I32 status, const std::string& body) {
  if (responded_ || connection_ == nullptr) {
    return false;
  }
  responded_ = true;
  std::string response;
  response.reserve(body.size() + 128);
  response.append("HTTP/1.1 ");
  response.append(std::to_string(status));
  response.push_back(' ');
  response.append(ReasonPhrase(status));
  response.append("\r\nContent-Type: application/json; charset=utf-8\r\n");
  response.append("Content-Length: ");
  response.append(std::to_string(body.size()));
  response.append("\r\nConnection: close\r\n\r\n");
  response.append(body);
  return connection_->Send(response);
}

Bool HttpExchange::BeginSse() {
  if (responded_ || connection_ == nullptr) {
    return false;
  }
  responded_ = true;
  sse_ = true;
  const std::string headers =
      "HTTP/1.1 200 OK\r\n"
      "Content-Type: text/event-stream\r\n"
      "Cache-Control: no-cache\r\n"
      "Connection: close\r\n"
      "Transfer-Encoding: chunked\r\n\r\n";
  return connection_->Send(headers);
}

Bool HttpExchange::SendSseEvent(const std::string& data) {
  if (!sse_ || connection_ == nullptr) {
    return false;
  }
  std::string payload = "data: ";
  payload.append(data);
  payload.append("\n\n");
  C8 header[32] = {0};
  std::snprintf(header, sizeof(header), "%llx\r\n",
                static_cast<unsigned long long>(payload.size()));
  if (!connection_->Send(header)) {
    return false;
  }
  if (!connection_->Send(payload)) {
    return false;
  }
  return connection_->Send("\r\n");
}

Bool HttpExchange::EndSse() {
  if (!sse_ || connection_ == nullptr) {
    return false;
  }
  sse_ = false;
  return connection_->Send("0\r\n\r\n");
}

Bool HttpExchange::client_connected() const {
  return connection_ != nullptr && connection_->connected.load();
}

// HTTP 服务器内部实现。
struct HttpServer::Impl {
  // 监听 socket。
  SocketHandle listener = kInvalidSocket;

  // accept 线程。
  std::thread accept_thread;

  // 是否运行。
  std::atomic<Bool> running{false};

  // 是否停止。
  std::atomic<Bool> stopping{false};

  // 是否已经执行过停止流程（保证 Stop 幂等）。
  Bool stopped_ = false;

  // Windows WSAStartup 是否已成功且尚未 WSACleanup。
  Bool wsa_started_ = false;

  // 实际端口。
  U16 port = 0;

  // 最大连接数。
  U32 max_connections = 0;

  // 请求体上限。
  U32 max_body_bytes = 0;

  // 读取超时。
  U32 read_timeout_ms = 0;

  // 处理器。
  HttpHandler handler;

  // 连接与线程管理锁。
  std::mutex mutex;

  // 活跃连接。
  std::unordered_map<SocketHandle, std::shared_ptr<HttpConnection>>
      connections;

  // 工作线程及其完成标记。
  struct Worker {
    std::thread thread;
    std::shared_ptr<std::atomic<Bool>> done;
  };

  // 工作线程列表。
  std::vector<Worker> workers;

  // 启动监听。
  // 释放 Winsock 引用；仅当成功 Startup 且尚未 Cleanup 时执行。
  void CleanupWsa() {
#if defined(_WIN32)
    if (wsa_started_) {
      WSACleanup();
      wsa_started_ = false;
    }
#endif
  }

  Status Start(const std::string& host, U16 requested_port,
               U32 max_conn, U32 body_limit, U32 timeout_ms,
               HttpHandler request_handler) {
    if (running.load()) {
      return MakeStatus(ErrorCode::kStateError,
                        "HttpServer: already running");
    }
    stopped_ = false;
#if defined(_WIN32)
    WSADATA data;
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
      stopped_ = true;
      return MakeStatus(ErrorCode::kIoError,
                        "HttpServer: WSAStartup failed");
    }
    wsa_started_ = true;
#else
    signal(SIGPIPE, SIG_IGN);
#endif
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    addrinfo* result = nullptr;
    const std::string port_text = std::to_string(requested_port);
    const I32 resolve = getaddrinfo(host.empty() ? nullptr : host.c_str(),
                                    port_text.c_str(), &hints, &result);
    if (resolve != 0 || result == nullptr) {
      CleanupWsa();
      stopped_ = true;
      return MakeStatus(ErrorCode::kIoError,
                        "HttpServer: getaddrinfo failed");
    }

    SocketHandle socket = kInvalidSocket;
    for (addrinfo* entry = result; entry != nullptr;
         entry = entry->ai_next) {
      socket = static_cast<SocketHandle>(::socket(
          entry->ai_family, entry->ai_socktype, entry->ai_protocol));
      if (socket == kInvalidSocket) {
        continue;
      }
      I32 reuse = 1;
      setsockopt(socket, SOL_SOCKET, SO_REUSEADDR,
                 reinterpret_cast<const C8*>(&reuse), sizeof(reuse));
      if (::bind(socket, entry->ai_addr,
                 static_cast<I32>(entry->ai_addrlen)) == 0 &&
          ::listen(socket, 32) == 0) {
        break;
      }
      CloseSocket(socket);
      socket = kInvalidSocket;
    }
    freeaddrinfo(result);
    if (socket == kInvalidSocket) {
      CleanupWsa();
      stopped_ = true;
      return MakeStatus(ErrorCode::kIoError,
                        "HttpServer: bind/listen failed");
    }

    U16 actual_port = requested_port;
    if (requested_port == 0) {
      sockaddr_storage address{};
      SocketLength length = static_cast<SocketLength>(sizeof(address));
      if (getsockname(socket, reinterpret_cast<sockaddr*>(&address),
                      &length) == 0) {
        if (address.ss_family == AF_INET) {
          actual_port = ntohs(
              reinterpret_cast<sockaddr_in*>(&address)->sin_port);
        } else if (address.ss_family == AF_INET6) {
          actual_port = ntohs(
              reinterpret_cast<sockaddr_in6*>(&address)->sin6_port);
        }
      }
    }

    listener = socket;
    port = actual_port;
    max_connections = max_conn;
    max_body_bytes = body_limit;
    read_timeout_ms = timeout_ms;
    handler = std::move(request_handler);
    stopping.store(false);
    running.store(true);
    try {
      // 捕获局部 socket，避免 Stop 同时修改 listener 成员造成数据竞争。
      accept_thread = std::thread([this, socket]() { AcceptLoop(socket); });
    } catch (...) {
      CloseSocket(socket);
      listener = kInvalidSocket;
      running.store(false);
      stopping.store(true);
      stopped_ = true;
      CleanupWsa();
      return MakeStatus(ErrorCode::kInternal,
                        "HttpServer: cannot start accept thread");
    }
    return Status::Ok();
  }

  // 停止服务器。
  void Stop() {
    if (stopped_) {
      CleanupWsa();
      return;
    }
    stopped_ = true;
    if (!running.load() && !stopping.load()) {
      CleanupWsa();
      return;
    }
    stopping.store(true);
    running.store(false);
    if (listener != kInvalidSocket) {
      ShutdownSocket(listener);
      CloseSocket(listener);
      listener = kInvalidSocket;
    }
    if (accept_thread.joinable()) {
      accept_thread.join();
    }

    std::vector<Worker> pending_workers;
    {
      std::lock_guard<std::mutex> lock(mutex);
      for (auto& entry : connections) {
        ShutdownSocket(entry.first);
      }
      pending_workers = std::move(workers);
      workers.clear();
    }
    for (Worker& worker : pending_workers) {
      if (worker.thread.joinable()) {
        worker.thread.join();
      }
    }
    {
      std::lock_guard<std::mutex> lock(mutex);
      connections.clear();
    }
    CleanupWsa();
  }

  // accept 循环。
  //
  // 参数：
  //   listen_socket：监听 socket 的本地副本；Stop 关闭监听时不会产生
  //   成员读写竞争。
  void AcceptLoop(SocketHandle listen_socket) {
    while (!stopping.load()) {
      sockaddr_storage address{};
      SocketLength length = static_cast<SocketLength>(sizeof(address));
      const SocketHandle client = static_cast<SocketHandle>(::accept(
          listen_socket, reinterpret_cast<sockaddr*>(&address), &length));
      if (client == kInvalidSocket) {
        if (stopping.load()) {
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        continue;
      }
      SetSocketTimeout(client, read_timeout_ms);

      auto connection = std::make_shared<HttpConnection>();
      connection->socket = client;
      auto done = std::make_shared<std::atomic<Bool>>(false);

      Bool reject = false;
      std::vector<Worker> finished;
      {
        std::lock_guard<std::mutex> lock(mutex);
        // 移出已完成线程，锁外 join，避免与线程退出时的 map 清理死锁。
        for (auto iterator = workers.begin(); iterator != workers.end();) {
          if (iterator->done->load()) {
            finished.push_back(std::move(*iterator));
            iterator = workers.erase(iterator);
          } else {
            ++iterator;
          }
        }
        if (connections.size() >= max_connections) {
          reject = true;
        } else {
          connections.emplace(client, connection);
          Worker worker;
          worker.done = done;
          try {
            worker.thread = std::thread([this, connection, done]() {
              HandleConnection(connection, done);
            });
            workers.push_back(std::move(worker));
          } catch (...) {
            connections.erase(client);
            reject = true;
          }
        }
      }
      for (Worker& worker : finished) {
        if (worker.thread.joinable()) {
          worker.thread.join();
        }
      }
      if (reject) {
        SendDirectError(connection, 503, "too many connections",
                        "server_busy");
        ShutdownSocket(client);
        CloseSocket(client);
      }
    }
  }

  // 处理单个连接。
  void HandleConnection(std::shared_ptr<HttpConnection> connection,
                        std::shared_ptr<std::atomic<Bool>> done) {
    HttpRequest request;
    std::string message;
    const ReadResult read_result = ReadRequest(connection, &request, &message);
    if (read_result == ReadResult::kOk) {
      HttpExchange exchange(connection, std::move(request));
      Bool handled = false;
      if (handler) {
        handled = handler(&exchange);
      }
      if (!handled && !exchange.responded()) {
        exchange.SendJson(500, BuildErrorBody("internal error",
                                               "internal_error"));
      }
    } else if (read_result == ReadResult::kBadRequest) {
      SendDirectError(connection, 400,
                      message.empty() ? "bad request" : message,
                      "invalid_request");
    } else if (read_result == ReadResult::kTooLarge) {
      SendDirectError(connection, 413, "request body too large",
                      "body_too_large");
    } else if (read_result == ReadResult::kTimeout) {
      SendDirectError(connection, 408, "request read timeout",
                      "request_timeout");
    }

    done->store(true);
    // 先从活跃表移除，再关闭 socket，避免 Stop 对已关闭/复用的句柄操作。
    {
      std::lock_guard<std::mutex> lock(mutex);
      connections.erase(connection->socket);
    }
    ShutdownSocket(connection->socket);
    CloseSocket(connection->socket);
  }

  // 读取并解析请求。
  ReadResult ReadRequest(
      const std::shared_ptr<HttpConnection>& connection,
      HttpRequest* request, std::string* message) {
    std::string buffer;
    Usize header_end = 0;
    while (true) {
      header_end = buffer.find("\r\n\r\n");
      if (header_end != std::string::npos) {
        header_end += 4;
        if (header_end > kMaxHeaderBytes) {
          *message = "header too large";
          return ReadResult::kBadRequest;
        }
        break;
      }
      if (buffer.size() > kMaxHeaderBytes) {
        *message = "header too large";
        return ReadResult::kBadRequest;
      }
      C8 chunk[kReadBufferSize] = {0};
      const I32 received =
          RecvBytes(connection->socket, chunk, kReadBufferSize);
      if (received <= 0) {
        if (received < 0 && IsReceiveTimeout()) {
          return ReadResult::kTimeout;
        }
        return ReadResult::kClosed;
      }
      buffer.append(chunk, static_cast<Usize>(received));
    }

    const std::string header_block = buffer.substr(0, header_end - 4);
    Usize line_begin = 0;
    Bool first_line = true;
    U64 content_length = 0;
    Bool has_content_length = false;
    Bool expect_continue = false;
    while (line_begin < header_block.size()) {
      const Usize line_end = header_block.find("\r\n", line_begin);
      const std::string line = header_block.substr(
          line_begin, line_end == std::string::npos
                          ? std::string::npos
                          : line_end - line_begin);
      if (first_line) {
        first_line = false;
        const Usize first_space = line.find(' ');
        const Usize second_space =
            first_space == std::string::npos
                ? std::string::npos
                : line.find(' ', first_space + 1);
        if (first_space == std::string::npos ||
            second_space == std::string::npos) {
          *message = "invalid request line";
          return ReadResult::kBadRequest;
        }
        request->method = line.substr(0, first_space);
        const std::string target =
            line.substr(first_space + 1, second_space - first_space - 1);
        const Usize question = target.find('?');
        request->path = question == std::string::npos
                            ? target
                            : target.substr(0, question);
        request->query = question == std::string::npos
                             ? std::string()
                             : target.substr(question + 1);
      } else {
        const Usize colon = line.find(':');
        if (colon != std::string::npos) {
          const std::string name =
              Lowercase(Trim(line.substr(0, colon)));
          const std::string value = Trim(line.substr(colon + 1));
          request->headers[name] = value;
          if (name == "content-length") {
            C8* end = nullptr;
            const unsigned long long parsed =
                std::strtoull(value.c_str(), &end, 10);
            if (end == value.c_str() || *end != '\0') {
              *message = "invalid content-length";
              return ReadResult::kBadRequest;
            }
            content_length = parsed;
            has_content_length = true;
          } else if (name == "transfer-encoding") {
            *message = "chunked request body is not supported";
            return ReadResult::kBadRequest;
          } else if (name == "expect") {
            expect_continue =
                Lowercase(value).find("100-continue") != std::string::npos;
          }
        }
      }
      if (line_end == std::string::npos) {
        break;
      }
      line_begin = line_end + 2;
    }

    if (has_content_length &&
        content_length > static_cast<U64>(max_body_bytes)) {
      return ReadResult::kTooLarge;
    }
    if (request->method.empty()) {
      *message = "empty method";
      return ReadResult::kBadRequest;
    }
    // 支持 curl 等客户端的 Expect: 100-continue，避免其等待 1 s 后才发 body。
    if (expect_continue) {
      if (!connection->Send("HTTP/1.1 100 Continue\r\n\r\n")) {
        return ReadResult::kClosed;
      }
    }

    std::string body = buffer.substr(header_end);
    if (body.size() > content_length) {
      body.resize(static_cast<Usize>(content_length));
    }
    while (body.size() < content_length) {
      C8 chunk[kReadBufferSize] = {0};
      const Usize want = static_cast<Usize>(
          std::min<U64>(kReadBufferSize, content_length - body.size()));
      const I32 received = RecvBytes(connection->socket, chunk, want);
      if (received <= 0) {
        if (received < 0 && IsReceiveTimeout()) {
          return ReadResult::kTimeout;
        }
        return ReadResult::kClosed;
      }
      body.append(chunk, static_cast<Usize>(received));
    }
    request->body = std::move(body);
    ParseQuery(request->query, &request->query_params);
    return ReadResult::kOk;
  }
};

HttpServer::HttpServer() : impl_(std::make_unique<Impl>()) {}

HttpServer::~HttpServer() {
  if (impl_ != nullptr) {
    impl_->Stop();
  }
}

Status HttpServer::Start(const std::string& host, U16 port,
                         U32 max_connections, U32 max_body_bytes,
                         U32 read_timeout_ms, HttpHandler handler) {
  if (impl_ == nullptr) {
    impl_ = std::make_unique<Impl>();
  }
  if (host.empty() || max_connections == 0 || max_body_bytes == 0 ||
      read_timeout_ms == 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "HttpServer::Start: invalid argument");
  }
  if (!handler) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "HttpServer::Start: handler must not be empty");
  }
  return impl_->Start(host, port, max_connections, max_body_bytes,
                      read_timeout_ms, std::move(handler));
}

Status HttpServer::Stop() {
  if (impl_ != nullptr) {
    impl_->Stop();
  }
  return Status::Ok();
}

U16 HttpServer::port() const {
  return impl_ == nullptr ? 0 : impl_->port;
}

Bool HttpServer::running() const {
  return impl_ != nullptr && impl_->running.load();
}

}  // namespace service
}  // namespace mobilenet
