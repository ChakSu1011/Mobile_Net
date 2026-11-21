// 模块：Session 日志工具
//
// 职责：
//   提供线程安全的级别日志，供 HTTP 服务器、ChatService 与会话管理使用。
//
// 边界：
//   - 只输出到 stderr；不写文件、不访问网络；
//   - 不记录 prompt/token 内容，避免隐私泄漏；
//   - 日志级别为全局状态，进程内共享。
//
// 依赖：
//   C++17 标准库、Utils/cpp/types.h。

#ifndef MOBILENET_SERVICE_SESSION_LOG_H_
#define MOBILENET_SERVICE_SESSION_LOG_H_

#include <string>

#include "Utils/cpp/types.h"

namespace mobilenet {
namespace service {

// 日志级别。
enum class LogLevel {
  kError = 0,
  kWarn = 1,
  kInfo = 2,
  kDebug = 3,
};

// 设置全局日志级别。
//
// 参数：
//   level：error/warn/info/debug；无法识别时保持 info。
void SetLogLevel(const std::string& level);

// 返回当前日志级别。
LogLevel CurrentLogLevel();

// 输出一条日志。
//
// 参数：
//   level：日志级别；
//   message：消息文本。
void LogMessage(LogLevel level, const std::string& message);

// 便捷函数。
void LogError(const std::string& message);
void LogWarn(const std::string& message);
void LogInfo(const std::string& message);
void LogDebug(const std::string& message);

}  // namespace service
}  // namespace mobilenet

#endif  // MOBILENET_SERVICE_SESSION_LOG_H_
