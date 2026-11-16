// 模块：Runtime 资源释放栈实现
//
// 职责：
//   实现 ResourceStack 的登记、逆序释放与幂等清空；加载失败回滚与正常
//   Destroy 共用本实现。
//
// 边界：
//   - 只执行调用方登记的释放闭包，不判断资源类型；
//   - 单个闭包失败不中断后续释放，最终返回首个错误；
//   - 不保证线程安全，只允许单线程使用。
//
// 依赖：
//   Runtime/common/resource_stack.h、C++17 标准库。

#include "Runtime/common/resource_stack.h"

#include <utility>

namespace mobilenet {
namespace runtime {

void ResourceStack::Push(std::string label, Release release) {
  entries_.push_back(Entry{std::move(label), std::move(release)});
}

Status ResourceStack::Clear() {
  Status first_error = Status::Ok();
  while (!entries_.empty()) {
    Entry entry = std::move(entries_.back());
    entries_.pop_back();
    if (!entry.release) {
      continue;
    }
    Status status = entry.release();
    if (!status.ok() && first_error.ok()) {
      first_error = std::move(status);
    }
  }
  return first_error;
}

std::size_t ResourceStack::size() const { return entries_.size(); }

}  // namespace runtime
}  // namespace mobilenet
