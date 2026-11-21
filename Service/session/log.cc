// 模块：Session 日志工具实现
//
// 职责：
//   实现级别过滤、时间戳与线程安全输出。
//
// 边界：
//   只写 stderr；消息内容由调用方保证不含敏感 token。

#include "Service/session/log.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <string>

namespace mobilenet {
namespace service {
namespace {

// 全局日志级别；默认 info。
std::atomic<LogLevel> g_level{LogLevel::kInfo};

// 输出互斥锁，避免多行日志交错。
std::mutex g_mutex;

// 返回级别名称。
const C8* LevelName(LogLevel level) {
  switch (level) {
    case LogLevel::kError:
      return "error";
    case LogLevel::kWarn:
      return "warn";
    case LogLevel::kInfo:
      return "info";
    case LogLevel::kDebug:
      return "debug";
  }
  return "info";
}

}  // namespace

void SetLogLevel(const std::string& level) {
  if (level == "error") {
    g_level.store(LogLevel::kError);
  } else if (level == "warn") {
    g_level.store(LogLevel::kWarn);
  } else if (level == "debug") {
    g_level.store(LogLevel::kDebug);
  } else {
    g_level.store(LogLevel::kInfo);
  }
}

LogLevel CurrentLogLevel() { return g_level.load(); }

void LogMessage(LogLevel level, const std::string& message) {
  if (static_cast<I32>(level) > static_cast<I32>(g_level.load())) {
    return;
  }
  const auto now = std::chrono::system_clock::now();
  const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
  const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now.time_since_epoch()) %
                      1000;
  std::tm local_time{};
#if defined(_WIN32)
  localtime_s(&local_time, &seconds);
#else
  localtime_r(&seconds, &local_time);
#endif
  C8 buffer[32] = {0};
  std::strftime(buffer, sizeof(buffer), "%H:%M:%S", &local_time);
  std::lock_guard<std::mutex> lock(g_mutex);
  std::fprintf(stderr, "[%s.%03lld] [%s] %s\n", buffer,
               static_cast<long long>(millis.count()), LevelName(level),
               message.c_str());
  std::fflush(stderr);
}

void LogError(const std::string& message) {
  LogMessage(LogLevel::kError, message);
}

void LogWarn(const std::string& message) {
  LogMessage(LogLevel::kWarn, message);
}

void LogInfo(const std::string& message) {
  LogMessage(LogLevel::kInfo, message);
}

void LogDebug(const std::string& message) {
  LogMessage(LogLevel::kDebug, message);
}

}  // namespace service
}  // namespace mobilenet
