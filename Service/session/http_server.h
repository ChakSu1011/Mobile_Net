// 模块：Session 最小 HTTP/1.1 服务器
//
// 职责：
//   监听 TCP、解析 HTTP/1.1 请求、提供普通 JSON 响应与 SSE chunked 响应。
//
// 边界：
//   - 只支持 GET/POST/DELETE、Content-Length 请求体、单请求单连接；
//   - 不支持 HTTPS、HTTP/2、WebSocket、chunked 请求体；
//   - 不解析业务 JSON；由上层 handler 负责；
//   - 线程安全：Start/Stop 互斥；连接并发处理。
//
// 依赖：
//   C++17 socket（Winsock/POSIX）、Utils/cpp/types.h。

#ifndef MOBILENET_SERVICE_SESSION_HTTP_SERVER_H_
#define MOBILENET_SERVICE_SESSION_HTTP_SERVER_H_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace service {

// HTTP 请求。
struct HttpRequest {
  // 方法（大写）。
  std::string method;

  // 路径（不含查询串）。
  std::string path;

  // 原始查询串。
  std::string query;

  // 请求体。
  std::string body;

  // 头部；键为小写。
  std::unordered_map<std::string, std::string> headers;

  // 路径参数。
  std::unordered_map<std::string, std::string> params;

  // 查询参数。
  std::unordered_map<std::string, std::string> query_params;
};

// 内部连接对象；定义在 http_server.cc。
struct HttpConnection;

// HTTP 交换对象。
//
// 功能：
//   向连接写普通响应或 SSE 事件；状态由连接线程管理。
//
// 边界：
//   - 每个请求只允许一种响应模式；普通响应与 SSE 互斥；
//   - 写失败后 client_connected() 返回 false；
//   - 不拥有 socket 生命周期。
class HttpExchange {
 public:
  // 构造器仅供 HttpServer 使用。
  HttpExchange(std::shared_ptr<HttpConnection> connection,
               HttpRequest request);

  ~HttpExchange();

  HttpExchange(const HttpExchange&) = delete;
  HttpExchange& operator=(const HttpExchange&) = delete;

  // 返回请求。
  const HttpRequest& request() const { return request_; }

  // 添加路径参数。
  void AddParam(const std::string& key, const std::string& value);

  // 发送普通 JSON 响应。
  //
  // 参数：
  //   status：HTTP 状态码；
  //   body：JSON 文本。
  // 返回：
  //   成功返回 true；已发送过响应或写失败返回 false。
  Bool SendJson(I32 status, const std::string& body);

  // 开始 SSE 响应，写 200 与 chunked 头。
  Bool BeginSse();

  // 发送一个 SSE 事件：`data: <data>\n\n`。
  Bool SendSseEvent(const std::string& data);

  // 结束 SSE chunked 响应。
  Bool EndSse();

  // 返回客户端是否仍连接。
  Bool client_connected() const;

  // 返回是否已开始响应。
  Bool responded() const { return responded_; }

 private:
  // 连接。
  std::shared_ptr<HttpConnection> connection_;

  // 请求。
  HttpRequest request_;

  // 是否已写响应。
  Bool responded_ = false;

  // 是否已进入 SSE 模式。
  Bool sse_ = false;
};

// HTTP 处理函数。
using HttpHandler = std::function<Bool(HttpExchange*)>;

// HTTP 服务器。
class HttpServer {
 public:
  HttpServer();

  ~HttpServer();

  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  // 启动监听与 accept 线程。
  //
  // 参数：
  //   host：监听地址；
  //   port：端口；0 表示系统分配；
  //   max_connections：最大并发连接；
  //   max_body_bytes：请求体上限；
  //   read_timeout_ms：套接字读取超时；
  //   handler：请求处理函数。
  // 返回：
  //   成功返回 kOk；socket/bind/listen 失败返回 kIoError/kInvalidArgument。
  Status Start(const std::string& host, U16 port, U32 max_connections,
               U32 max_body_bytes, U32 read_timeout_ms,
               HttpHandler handler);

  // 停止服务器；幂等。
  Status Stop();

  // 返回实际监听端口。
  U16 port() const;

  // 返回是否运行中。
  Bool running() const;

 private:
  // 内部实现。
  struct Impl;

  // PIMPL。
  std::unique_ptr<Impl> impl_;
};

}  // namespace service
}  // namespace mobilenet

#endif  // MOBILENET_SERVICE_SESSION_HTTP_SERVER_H_
