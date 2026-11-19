// 模块：Backend 注册表实现
//
// 职责：
//   实现进程内后端名称到工厂的注册、查询、创建与枚举。
//
// 边界：
//   不创建 Buffer/Kernel、不初始化设备；工厂在锁外执行；不持有 Backend。
//
// 依赖：
//   Runtime/backend/backend_registry.h、Runtime/common/status.h。

#include "Runtime/backend/backend_registry.h"

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include "Utils/cpp/types.h"

namespace mobilenet {

BackendRegistry& BackendRegistry::Instance() {
  static BackendRegistry registry;
  return registry;
}

Status BackendRegistry::Register(const std::string& name,
                                 BackendFactory factory) {
  if (name.empty()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "BackendRegistry::Register: name must not be empty");
  }
  if (!factory) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "BackendRegistry::Register: factory must not be empty: " + name);
  }
  std::lock_guard<std::mutex> lock(mutex_);
  const auto inserted =
      factories_.emplace(name, std::move(factory));
  if (!inserted.second) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "BackendRegistry::Register: duplicated backend: " +
                          name);
  }
  return Status::Ok();
}

Status BackendRegistry::Create(const std::string& name,
                               const BackendConfig& config,
                               std::unique_ptr<Backend>* backend) const {
  if (backend == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "BackendRegistry::Create: backend output must not be null");
  }
  BackendFactory factory;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = factories_.find(name);
    if (it == factories_.end()) {
      return MakeStatus(ErrorCode::kUnsupportedBackend,
                        "BackendRegistry::Create: unknown backend: " + name);
    }
    factory = it->second;
  }
  // 工厂在锁外执行，避免初始化设备时长时间持锁或重入注册表。
  return factory(config, backend);
}

std::vector<std::string> BackendRegistry::Names() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::string> names;
  names.reserve(factories_.size());
  for (const auto& entry : factories_) {
    names.push_back(entry.first);
  }
  return names;
}

BackendRegistrar::BackendRegistrar(const C8* name, BackendFactory factory) {
  // 静态初始化阶段无法向调用方返回错误；失败由注册测试或启动诊断发现。
  const Status status = BackendRegistry::Instance().Register(
      name != nullptr ? std::string(name) : std::string(), std::move(factory));
  (void)status;
}

}  // namespace mobilenet
