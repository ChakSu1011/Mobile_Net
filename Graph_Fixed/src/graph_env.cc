// 模块：Runtime Graph GraphEnv 实现
//
// 职责：
//   实现 graph 级共享 Storage 的注册与查询。
//
// 边界：
//   只维护元信息表，不分配共享 buffer，不做线程同步。
//
// 依赖：
//   runtime_graph/graph_env.h。

#include "runtime_graph/graph_env.h"

#include <map>
#include <string>

#include "Runtime/common/status.h"

namespace mobilenet {
namespace runtime_graph {

Status GraphEnv::AddSharedStorage(const std::string& name,
                                  const Storage& storage) {
  if (name.empty()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "GraphEnv::AddSharedStorage: name must not be empty");
  }
  const auto inserted = shared_storages_.emplace(name, storage);
  if (!inserted.second) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "GraphEnv::AddSharedStorage: duplicate storage name: " +
                          name);
  }
  return Status::Ok();
}

Storage* GraphEnv::GetSharedStorage(const std::string& name) {
  const auto iterator = shared_storages_.find(name);
  return iterator == shared_storages_.end() ? nullptr : &iterator->second;
}

const Storage* GraphEnv::GetSharedStorage(const std::string& name) const {
  const auto iterator = shared_storages_.find(name);
  return iterator == shared_storages_.end() ? nullptr : &iterator->second;
}

const std::map<std::string, Storage>& GraphEnv::SharedStorages() const {
  return shared_storages_;
}

}  // namespace runtime_graph
}  // namespace mobilenet
