// 模块：Session OpenAPI 描述
//
// 职责：
//   提供已实现 HTTP 端点的 OpenAPI 3.0.3 静态 JSON 描述。
//
// 边界：
//   - 静态文档，不反射运行时路由；
//   - 只描述 P0 端点；
//   - 不包含鉴权/TLS 描述（基础版未实现）。
//
// 依赖：
//   C++17 标准库、Utils/cpp/types.h。

#ifndef MOBILENET_SERVICE_SESSION_OPENAPI_SPEC_H_
#define MOBILENET_SERVICE_SESSION_OPENAPI_SPEC_H_

#include "Utils/cpp/types.h"

namespace mobilenet {
namespace service {

// 返回 OpenAPI 3.0.3 JSON。
//
// 返回：
//   以 NUL 结尾的合法 JSON 字符串；生命周期为进程级。
const C8* OpenApiSpecJson();

}  // namespace service
}  // namespace mobilenet

#endif  // MOBILENET_SERVICE_SESSION_OPENAPI_SPEC_H_
