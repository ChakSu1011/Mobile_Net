// 模块：Runtime GraphEnv
//
// 职责：
//   保存 graph 级全局共享的环境资源元信息，例如整个 graph 只需要一份的
//   共享 buffer Storage；Manager 通过 GraphEnv 遍历并统一规划 workspace。
//
// 边界：
//   - 只保存 Storage 元信息，不拥有也不分配 data_ptr 指向的内存；
//   - 不保存 OpEnv、kernel 句柄或设备上下文；
//   - 生命周期由 Graph 通过 unique_ptr 持有，Engine 初始化阶段创建。
//
// 依赖：
//   Runtime/common/status.h、storage.h、C++17 标准库。

#ifndef MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_GRAPH_ENV_H_
#define MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_GRAPH_ENV_H_

#include <map>
#include <string>

#include "Runtime/common/status.h"
#include "storage.h"

namespace mobilenet {
namespace runtime_graph {

// GraphEnv 保存 graph 级共享资源。
//
// 功能：
//   以名称为键保存共享 Storage，供 OpEnv 通过 GraphEnv 指针访问，供 Manager
//   统一回填 workspace 地址。
//
// 边界：
//   - 同一名称只能注册一次；
//   - 不负责共享 buffer 的申请、释放和地址回填；
//   - 不保证线程安全，初始化阶段由单线程使用。
class GraphEnv {
 public:
  GraphEnv() = default;
  ~GraphEnv() = default;

  GraphEnv(const GraphEnv&) = delete;
  GraphEnv& operator=(const GraphEnv&) = delete;
  GraphEnv(GraphEnv&&) = delete;
  GraphEnv& operator=(GraphEnv&&) = delete;

  // 注册一份 graph 级共享 storage。
  //
  // 参数：
  //   name：共享资源名称，不允许为空；
  //   storage：共享资源的 storage 描述。
  // 返回：
  //   成功返回 OK；名称为空或重复注册返回 INVALID_ARGUMENT。
  Status AddSharedStorage(const std::string& name, const Storage& storage);

  // 按名称查找共享 storage。
  //
  // 参数：
  //   name：共享资源名称。
  // 返回：
  //   找到时返回可修改指针；不存在时返回 nullptr。
  Storage* GetSharedStorage(const std::string& name);

  // 按名称查找共享 storage（只读）。
  //
  // 参数：
  //   name：共享资源名称。
  // 返回：
  //   找到时返回只读指针；不存在时返回 nullptr。
  const Storage* GetSharedStorage(const std::string& name) const;

  // 返回全部共享 storage，供 Manager 遍历规划。
  const std::map<std::string, Storage>& SharedStorages() const;

 private:
  std::map<std::string, Storage> shared_storages_;
};

}  // namespace runtime_graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_GRAPH_ENV_H_
