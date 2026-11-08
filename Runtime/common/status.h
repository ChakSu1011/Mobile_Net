// 模块：Runtime 公共状态码与 Status 返回值
//
// 职责：
//   为 Runtime 各模块（Kernel、Backend、Graph、Loader、Executor 等）提供统一的
//   错误码与轻量返回值类型，避免各模块重复定义错误语义。
//
// 边界：
//   本文件只定义错误码与状态对象，不包含任何张量、设备或图结构定义；
//   不抛出异常，所有可预期错误均通过 Status 返回。
//
// 依赖：
//   仅依赖 C++17 标准库。

#ifndef MOBILENET_RUNTIME_COMMON_STATUS_H_
#define MOBILENET_RUNTIME_COMMON_STATUS_H_

#include <string>
#include <string_view>
#include <utility>

namespace mobilenet {

// 错误码枚举。
//
// 取值约定：
//   - kOk 表示成功；
//   - 其余取值分别对应参数、状态、版本、缓冲区、设备、内核、内部错误等类别；
//   - 新增错误码必须同步更新 ErrorCodeName()。
enum class ErrorCode {
  kOk = 0,
  kInvalidArgument = 1,
  kStateError = 2,
  kUnsupportedOp = 3,
  kVersionMismatch = 4,
  kBufferError = 5,
  kDeviceError = 6,
  kKernelError = 7,
  kInternal = 8,
  kUnsupportedDtype = 9,
  kUnsupportedLayout = 10,
};

// 返回错误码的稳定字符串名称。
//
// 参数：
//   code：错误码。
// 返回：
//   形如 "INVALID_ARGUMENT" 的常量字符串；未知错误码返回 "UNKNOWN"。
const char* ErrorCodeName(ErrorCode code);

// 轻量状态对象。
//
// 功能：
//   表示一次操作成功或失败，并携带错误码与人类可读消息。
//
// 边界：
//   不携带调用栈、不分配错误链；消息只用于日志与测试定位。
class Status {
 public:
  // 构造成功状态。
  Status() = default;

  // 构造指定错误码与消息的状态。
  //
  // 参数：
  //   code：错误码，不允许为 kOk；
  //   message：错误描述，允许为空。
  Status(ErrorCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  // 构造成功状态。
  static Status Ok() { return Status(); }

  // 构造失败状态。
  static Status Error(ErrorCode code, std::string message) {
    return Status(code, std::move(message));
  }

  // 是否成功。
  bool ok() const { return code_ == ErrorCode::kOk; }

  // 错误码；成功时返回 kOk。
  ErrorCode code() const { return code_; }

  // 错误消息；成功时为空字符串。
  const std::string& message() const { return message_; }

 private:
  ErrorCode code_ = ErrorCode::kOk;
  std::string message_;
};

// 构造失败状态。
//
// 参数：
//   code：错误码；
//   message：错误描述。
// 返回：
//   Status 对象。
inline Status MakeStatus(ErrorCode code, std::string message) {
  return Status::Error(code, std::move(message));
}

// 给已有失败状态增加上下文前缀。
//
// 功能：
//   在模块边界补充“哪个阶段/哪个对象”失败，保留原始错误码。
//
// 参数：
//   status：原始状态；若成功则原样返回；
//   prefix：前缀文本，例如 "GraphBuilder::Finalize"。
// 返回：
//   带前缀的新状态；成功状态保持不变。
Status PrefixStatus(const Status& status, std::string_view prefix);

}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_COMMON_STATUS_H_
