// 模块：Session 服务总装
//
// 职责：
//   组装配置、Tokenizer、Engine、ChatService、SessionManager 与 HTTP
//   服务器；注册路由并实现各端点。
//
// 边界：
//   - 不实现推理/分词算法；
//   - 不解析 Tokenizer 资产之外的模型文件；
//   - Stop 负责按依赖顺序释放资源。
//
// 依赖：
//   server_config、engine_client、pending_registry、chat_service、
//   session_manager、http_server/router、openai_api、openapi_spec。

#ifndef MOBILENET_SERVICE_SESSION_SERVER_H_
#define MOBILENET_SERVICE_SESSION_SERVER_H_

#include <chrono>
#include <memory>
#include <string>

#include "Runtime/common/json.h"
#include "Runtime/common/status.h"
#include "Service/session/chat_service.h"
#include "Service/session/engine_client.h"
#include "Service/session/http_router.h"
#include "Service/session/http_server.h"
#include "Service/session/openai_api.h"
#include "Service/session/pending_registry.h"
#include "Service/session/server_config.h"
#include "Service/session/session_manager.h"
#include "Service/tokenizer/tokenizer.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace service {

// 服务实例。
class Server {
 public:
  Server();

  ~Server();

  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;

  // 初始化：加载 Tokenizer、Engine 与业务对象。
  //
  // 参数：
  //   config：服务配置；
  //   engine：可选注入的 Engine；为空时创建 RuntimeInferenceEngine。
  // 返回：
  //   成功返回 kOk；资产/引擎加载失败返回对应错误。
  Status Init(const ServerConfig& config,
              std::unique_ptr<IEngine> engine = nullptr);

  // 启动 HTTP 监听。
  Status Start();

  // 停止服务；幂等。
  Status Stop();

  // 返回实际监听端口。
  U16 port() const;

  // 返回是否运行中。
  Bool running() const;

  // 返回配置。
  const ServerConfig& config() const { return config_; }

 private:
  // 路由入口。
  Bool Handle(HttpExchange* exchange);

  // 各端点。
  Bool HandleHealth(HttpExchange* exchange);
  Bool HandleModels(HttpExchange* exchange);
  Bool HandleChatCompletions(HttpExchange* exchange);
  Bool HandleCreateSession(HttpExchange* exchange);
  Bool HandleGetSession(HttpExchange* exchange);
  Bool HandleDeleteSession(HttpExchange* exchange);
  Bool HandleSessionChat(HttpExchange* exchange);
  Bool HandleOpenApi(HttpExchange* exchange);

  // 执行一次聊天（无状态或有状态）。
  //
  // 参数：
  //   exchange：HTTP 交换；
  //   request：已解析请求；
  //   session_id：有状态时非空；
  //   stateful：是否走 SessionManager。
  Bool RunChat(HttpExchange* exchange, const ChatRequest& request,
                const std::string& session_id, Bool stateful);

  // 解析请求体 JSON。
  Bool ParseJsonBody(HttpExchange* exchange, runtime::JsonValue* body,
                     ApiError* error);

  // 发送 API 错误。
  void SendApiError(HttpExchange* exchange, const ApiError& error);

  // 把 Status 映射为 API 错误并发送。
  void SendStatusError(HttpExchange* exchange, const Status& status);

  // 注册路由。
  void RegisterRoutes();

  // 配置。
  ServerConfig config_;

  // Tokenizer。
  std::unique_ptr<Tokenizer> tokenizer_;

  // Engine。
  std::unique_ptr<IEngine> engine_;

  // 请求事件注册表。
  PendingRegistry registry_;

  // 对话服务。
  std::unique_ptr<ChatService> chat_service_;

  // 会话管理。
  std::unique_ptr<SessionManager> sessions_;

  // HTTP 服务器。
  HttpServer http_;

  // 路由。
  HttpRouter router_;

  // 启动时间。
  std::chrono::steady_clock::time_point started_at_;

  // 是否已初始化。
  Bool initialized_ = false;
};

}  // namespace service
}  // namespace mobilenet

#endif  // MOBILENET_SERVICE_SESSION_SERVER_H_
