// 模块：Mock 后端设备缓冲
//
// 职责：
//   用主机对齐内存模拟设备 Buffer；DevicePtr() 返回主机地址，供 Mock
//   Backend 的 memcpy 传输与 CPU Kernel 直接访问。
//
// 边界：
//   - 只提供内存拥有与地址访问，不负责初始化、同步或布局转换；
//   - 生命周期由 Backend::CreateBuffer/ConvertParam 创建后移交给 Runtime；
//   - 不暴露额外的设备语义；上层仍按 Buffer 接口使用。
//
// 依赖：
//   Runtime/backend/backend.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_BACKEND_MOCK_MOCK_BUFFER_H_
#define MOBILENET_RUNTIME_BACKEND_MOCK_MOCK_BUFFER_H_

#include <cstdint>

#include "Runtime/backend/backend.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace mock {

// 主机内存模拟的设备缓冲。
class MockBuffer final : public Buffer {
 public:
  // 分配指定字节数的主机内存。
  //
  // 参数：
  //   bytes：字节数，必须大于 0；
  //   alignment：对齐字节数，必须是 2 的幂；0 表示默认 64。
  // 异常：
  //   分配失败抛出 std::bad_alloc；构造函数由 Backend 捕获并转成错误码。
  MockBuffer(U64 bytes, U64 alignment);

  ~MockBuffer() override;

  MockBuffer(const MockBuffer&) = delete;
  MockBuffer& operator=(const MockBuffer&) = delete;
  MockBuffer(MockBuffer&&) = delete;
  MockBuffer& operator=(MockBuffer&&) = delete;

  // 返回主机地址，作为不透明设备句柄使用。
  void* DevicePtr() const override;

  // 返回缓冲字节数。
  U64 Bytes() const override;

  // 返回可读写的主机数据指针。
  //
  // 返回：
  //   非空指针；仅供 Mock 后端内部与 CPU Kernel 使用。
  void* Data() const;

 private:
  // 主机内存起始地址。
  void* data_ = nullptr;

  // 分配的字节数。
  U64 bytes_ = 0;
};

}  // namespace mock
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_MOCK_MOCK_BUFFER_H_
