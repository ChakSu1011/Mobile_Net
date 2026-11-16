// 模块：Mock 后端设备缓冲实现
//
// 职责：
//   实现对齐主机内存的分配与释放，以及 Buffer 接口的地址/大小访问。
//
// 边界：
//   不做数据初始化；不做传输；不处理设备同步。
//
// 依赖：
//   Runtime/backend/mock/mock_buffer.h、C++17 标准库。

#include "Runtime/backend/mock/mock_buffer.h"

#include <cstddef>
#include <cstdlib>
#include <new>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace mobilenet {
namespace mock {

MockBuffer::MockBuffer(uint64_t bytes, uint64_t alignment) : bytes_(bytes) {
  const size_t align =
      alignment == 0 ? static_cast<size_t>(64) : static_cast<size_t>(alignment);
#if defined(_WIN32)
  data_ = _aligned_malloc(static_cast<size_t>(bytes), align);
#else
  void* pointer = nullptr;
  if (posix_memalign(&pointer, align, static_cast<size_t>(bytes)) != 0) {
    pointer = nullptr;
  }
  data_ = pointer;
#endif
  if (data_ == nullptr) {
    throw std::bad_alloc();
  }
}

MockBuffer::~MockBuffer() {
#if defined(_WIN32)
  _aligned_free(data_);
#else
  std::free(data_);
#endif
  data_ = nullptr;
}

void* MockBuffer::DevicePtr() const { return data_; }

uint64_t MockBuffer::Bytes() const { return bytes_; }

void* MockBuffer::Data() const { return data_; }

}  // namespace mock
}  // namespace mobilenet
