// 模块：Session HTTP 路由实现
//
// 职责：
//   实现路径分段匹配、参数提取与 404/405 错误响应。
//
// 边界：
//   不解析请求体；不处理业务错误。

#include "Service/session/http_router.h"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Service/session/json_writer.h"

namespace mobilenet {
namespace service {
namespace {

// 拆分路径；忽略空段。
std::vector<std::string> SplitPath(const std::string& path) {
  std::vector<std::string> segments;
  Usize cursor = 0;
  while (cursor < path.size()) {
    while (cursor < path.size() && path[cursor] == '/') {
      ++cursor;
    }
    if (cursor >= path.size()) {
      break;
    }
    const Usize begin = cursor;
    while (cursor < path.size() && path[cursor] != '/') {
      ++cursor;
    }
    segments.push_back(path.substr(begin, cursor - begin));
  }
  return segments;
}

// 发送路由错误。
void SendRouterError(HttpExchange* exchange, I32 status,
                     const std::string& code,
                     const std::string& message) {
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
  exchange->SendJson(status, writer.str());
}

// 匹配路径模式。
Bool MatchSegments(const std::vector<std::string>& pattern,
                   const std::vector<std::string>& path,
                   std::unordered_map<std::string, std::string>* params) {
  if (pattern.size() != path.size()) {
    return false;
  }
  params->clear();
  for (Usize index = 0; index < pattern.size(); ++index) {
    const std::string& expected = pattern[index];
    const std::string& actual = path[index];
    if (!expected.empty() && expected[0] == ':') {
      if (actual.empty()) {
        return false;
      }
      params->emplace(expected.substr(1), actual);
      continue;
    }
    if (expected != actual) {
      return false;
    }
  }
  return true;
}

}  // namespace

HttpRouter::HttpRouter() = default;

void HttpRouter::Add(const std::string& method, const std::string& pattern,
                     HttpHandler handler) {
  Route route;
  route.method = method;
  route.segments = SplitPath(pattern);
  route.handler = std::move(handler);
  routes_.push_back(std::move(route));
}

Bool HttpRouter::Handle(HttpExchange* exchange) const {
  if (exchange == nullptr) {
    return false;
  }
  const std::vector<std::string> path = SplitPath(exchange->request().path);
  Bool method_mismatch = false;
  for (const Route& route : routes_) {
    std::unordered_map<std::string, std::string> params;
    if (!MatchSegments(route.segments, path, &params)) {
      continue;
    }
    if (route.method != exchange->request().method) {
      method_mismatch = true;
      continue;
    }
    for (const auto& entry : params) {
      exchange->AddParam(entry.first, entry.second);
    }
    return route.handler(exchange);
  }
  if (method_mismatch) {
    SendRouterError(exchange, 405, "method_not_allowed",
                    "method not allowed for this path");
  } else {
    SendRouterError(exchange, 404, "not_found", "route not found");
  }
  return false;
}

}  // namespace service
}  // namespace mobilenet
