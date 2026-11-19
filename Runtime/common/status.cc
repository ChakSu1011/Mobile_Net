// 模块：Runtime Status 实现
//
// 职责：
//   实现 ErrorCodeName() 与 PrefixStatus() 两个与平台无关的辅助函数。
//
// 边界：
//   不引入日志系统、不依赖任何设备后端。

#include "Runtime/common/status.h"

#include <string>
#include "Utils/cpp/types.h"

namespace mobilenet {

const C8* ErrorCodeName(ErrorCode code) {
  switch (code) {
    case ErrorCode::kOk:
      return "OK";
    case ErrorCode::kInvalidArgument:
      return "INVALID_ARGUMENT";
    case ErrorCode::kStateError:
      return "STATE_ERROR";
    case ErrorCode::kUnsupportedOp:
      return "UNSUPPORTED_OP";
    case ErrorCode::kVersionMismatch:
      return "VERSION_MISMATCH";
    case ErrorCode::kBufferError:
      return "BUFFER_ERROR";
    case ErrorCode::kDeviceError:
      return "DEVICE_ERROR";
    case ErrorCode::kKernelError:
      return "KERNEL_ERROR";
    case ErrorCode::kInternal:
      return "INTERNAL";
    case ErrorCode::kUnsupportedDtype:
      return "UNSUPPORTED_DTYPE";
    case ErrorCode::kUnsupportedLayout:
      return "UNSUPPORTED_LAYOUT";
    case ErrorCode::kInvalidModelFile:
      return "INVALID_MODEL_FILE";
    case ErrorCode::kIoError:
      return "IO_ERROR";
    case ErrorCode::kDeviceOom:
      return "DEVICE_OOM";
    case ErrorCode::kUnsupportedBackend:
      return "UNSUPPORTED_BACKEND";
    case ErrorCode::kBackendInitError:
      return "BACKEND_INIT_ERROR";
    case ErrorCode::kRequestNotFound:
      return "REQUEST_NOT_FOUND";
    case ErrorCode::kTimeout:
      return "TIMEOUT";
    case ErrorCode::kCancelled:
      return "CANCELLED";
  }
  return "UNKNOWN";
}

Status PrefixStatus(const Status& status, std::string_view prefix) {
  if (status.ok()) {
    return status;
  }
  std::string message;
  message.reserve(prefix.size() + 2 + status.message().size());
  message.append(prefix);
  message.append(": ");
  message.append(status.message());
  return Status(status.code(), std::move(message));
}

}  // namespace mobilenet
