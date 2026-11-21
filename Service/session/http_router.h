// 模块：Session HTTP 路由
//
// 职责：
//   按方法与路径模式分发请求；支持 `:param` 路径参数。
//
// 边界：
//   - 不做 JSON 解析；不实现通配符/正则路由；
//   - 路由未匹配时直接写 404/405 错误响应；
//   - 线程安全：注册完成后只读，可多连接并发匹配。
//
// 依赖：
//   http_server.h（HttpExchange/HttpRequest）。

#ifndef MOBILENET_SERVICE_SESSION_HTTP_ROUTER_H_
#define MOBILENET_SERVICE_SESSION_HTTP_ROUTER_H_

#include <functional>
#include <string>
#include <vector>

#include "Service/session/http_server.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace service {

// HTTP 路由表。
class HttpRouter {
 public:
  HttpRouter();

  // 注册路由。
  //
  // 参数：
  //   method：GET/POST/DELETE；
  //   pattern：以 '/' 开头的路径；`/:name` 表示参数；
  //   handler：处理函数；返回 false 表示未成功写响应。
  void Add(const std::string& method, const std::string& pattern,
           HttpHandler handler);

  // 分发请求。
  //
  // 参数：
  //   exchange：HTTP 交换对象。
  // 返回：
  //   找到处理函数并执行返回 true；否则写 404/405 并返回 false。
  Bool Handle(HttpExchange* exchange) const;

 private:
  // 单条路由。
  struct Route {
    // HTTP 方法。
    std::string method;

    // 路径模式分段。
    std::vector<std::string> segments;

    // 处理函数。
    HttpHandler handler;
  };

  // 路由列表；注册完成后只读。
  std::vector<Route> routes_;
};

}  // namespace service
}  // namespace mobilenet

#endif  // MOBILENET_SERVICE_SESSION_HTTP_ROUTER_H_
