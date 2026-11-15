// 模块：Runtime 资源释放栈
//
// 职责：
//   按登记顺序逆序释放设备缓冲、OpEnv、临时资源等；加载失败回滚与正常
//   Destroy 复用同一路径，保证“后创建先释放”和幂等。
//
// 边界：
//   - 只保存释放闭包与标签，不拥有资源本身；
//   - 不负责停线程、Synchronize、GraphEnv/Backend 的最终释放（由 Destroy
//     按固定阶段编排）；
//   - 不保证线程安全，只允许单线程使用。
//
// 依赖：
//   Runtime/common/status.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_COMMON_RESOURCE_STACK_H_
#define MOBILENET_RUNTIME_COMMON_RESOURCE_STACK_H_

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "Runtime/common/status.h"

namespace mobilenet {
namespace runtime {

// 资源释放栈。
//
// 功能：
//   Loader/KVCache 每创建一个需要显式释放的资源就 Push 一个释放闭包；
//   Destroy/回滚时调用 Clear，逆序执行。
//
// 边界：
//   - Push 的闭包必须捕获释放所需的最小状态，不得引用已析构对象；
//   - Clear 幂等：重复调用为空操作；
//   - 单个闭包失败不中断后续释放，最终返回首个错误。
class ResourceStack {
 public:
  // 释放闭包类型；返回该资源的释放状态。
  using Release = std::function<Status()>;

  // 登记一个释放动作。
  //
  // 参数：
  //   label：资源标签，用于错误信息与泄漏报告；
  //   release：释放闭包，不允许为空。
  void Push(std::string label, Release release);

  // 逆序执行并清空全部释放动作。
  //
  // 返回：
  //   全部成功返回 kOk；否则返回首个失败状态（后续释放仍执行）。
  Status Clear();

  // 返回当前登记的释放动作数量。
  std::size_t size() const;

 private:
  // 单条释放记录。
  struct Entry {
    std::string label;
    Release release;
  };

  std::vector<Entry> entries_;
};

}  // namespace runtime
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_COMMON_RESOURCE_STACK_H_
