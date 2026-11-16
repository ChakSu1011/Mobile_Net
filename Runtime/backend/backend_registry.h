// 模块：Backend 注册表
//
// 职责：
//   维护后端名称到工厂函数的映射；Loader 在加载期按 RuntimeConfig.backend
//   名称创建 Backend 实例。各后端目录通过 MOBILENET_REGISTER_BACKEND 在
//   静态初始化阶段自注册。
//
// 边界：
//   - 只负责名称→工厂的注册与查询，不创建 Buffer/Kernel，不初始化设备；
//   - 不决定默认后端，不读取 RuntimeConfig；
//   - 注册在静态初始化或单线程启动阶段完成；Create 可在加载期多线程调用；
//   - 公共头不出现厂商类型。
//
// 依赖：
//   Runtime/backend/backend.h、Runtime/common/status.h；C++17 标准库。

#ifndef MOBILENET_RUNTIME_BACKEND_BACKEND_REGISTRY_H_
#define MOBILENET_RUNTIME_BACKEND_BACKEND_REGISTRY_H_

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "Runtime/backend/backend.h"
#include "Runtime/common/status.h"

namespace mobilenet {

// 后端工厂函数。
//
// 参数：
//   config：后端初始化配置；
//   backend：输出参数，成功时写入非空实例。
// 返回：
//   成功返回 kOk；失败返回具体错误码，且不修改 backend。
using BackendFactory =
    std::function<Status(const BackendConfig& config,
                         std::unique_ptr<Backend>* backend)>;

// 后端注册表。
//
// 功能：
//   提供进程内唯一的名称→工厂映射；支持注册、按名称创建与枚举。
//
// 边界：
//   - 不持有 Backend 实例，只持有工厂；
//   - 同名重复注册被拒绝；名称不可为空；
//   - 线程安全：Register/Create/Names 内部加锁，工厂在锁外执行。
class BackendRegistry {
 public:
  // 返回进程内唯一注册表。
  //
  // 返回：
  //   非空单例引用；首次调用时构造。
  static BackendRegistry& Instance();

  BackendRegistry(const BackendRegistry&) = delete;
  BackendRegistry& operator=(const BackendRegistry&) = delete;
  BackendRegistry(BackendRegistry&&) = delete;
  BackendRegistry& operator=(BackendRegistry&&) = delete;

  // 注册一个后端工厂。
  //
  // 参数：
  //   name：后端名称，例如 "ocl"；不允许为空；
  //   factory：工厂函数，不允许为空。
  // 返回：
  //   成功返回 kOk；名称为空/工厂为空/名称重复返回 kInvalidArgument。
  Status Register(const std::string& name, BackendFactory factory);

  // 按名称创建后端实例。
  //
  // 参数：
  //   name：后端名称；
  //   config：传给工厂的配置；
  //   backend：输出参数，不允许为空；成功时写入实例。
  // 返回：
  //   成功返回 kOk；名称未注册返回 kUnsupportedBackend；backend 为空返回
  //   kInvalidArgument；工厂失败时透传其错误码。
  Status Create(const std::string& name, const BackendConfig& config,
                std::unique_ptr<Backend>* backend) const;

  // 返回已注册后端名称（按字典序）。
  std::vector<std::string> Names() const;

 private:
  // 只允许通过 Instance() 构造单例。
  BackendRegistry() = default;

  // 名称到工厂的映射。
  std::map<std::string, BackendFactory> factories_;

  // 保护 factories_ 的互斥量；Create 在锁外调用工厂。
  mutable std::mutex mutex_;
};

// 静态注册器。
//
// 功能：
//   在静态初始化阶段把工厂注册到 BackendRegistry；供
//   MOBILENET_REGISTER_BACKEND 宏使用。
//
// 边界：
//   - 构造函数不抛异常；注册失败（空名/重复）被忽略，由编译期/测试发现；
//   - 不拥有工厂或 Backend。
class BackendRegistrar {
 public:
  // 注册后端。
  //
  // 参数：
  //   name：后端名称，非空；
  //   factory：工厂函数，非空。
  BackendRegistrar(const char* name, BackendFactory factory);

  BackendRegistrar(const BackendRegistrar&) = delete;
  BackendRegistrar& operator=(const BackendRegistrar&) = delete;
  BackendRegistrar(BackendRegistrar&&) = delete;
  BackendRegistrar& operator=(BackendRegistrar&&) = delete;
};

// 拼接宏的中间层，保证 __LINE__ 在展开后再拼接。
#define MOBILENET_BACKEND_CONCAT_IMPL(lhs, rhs) lhs##rhs
#define MOBILENET_BACKEND_CONCAT(lhs, rhs) \
  MOBILENET_BACKEND_CONCAT_IMPL(lhs, rhs)

// 在全局/命名空间作用域注册后端。
//
// 用法：
//   MOBILENET_REGISTER_BACKEND("ocl", CreateOclBackend);
//
// 说明：
//   name 与 factory 在静态初始化阶段求值；factory 必须是可转换为
//   BackendFactory 的具名函数或函数对象，不要直接传含逗号的 lambda，
//   以免宏参数被拆分。注册失败不会终止进程。
#define MOBILENET_REGISTER_BACKEND(name, factory)                    \
  static ::mobilenet::BackendRegistrar                              \
      MOBILENET_BACKEND_CONCAT(mobilenet_backend_registrar_,         \
                               __LINE__)((name), (factory))

}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_BACKEND_REGISTRY_H_
