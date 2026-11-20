// 模块：OCL 设备缓冲实现
//
// 职责：
//   实现 Buffer 接口的地址/大小访问。
//
// 边界：
//   不释放 cl_mem；不记录子偏移。
//
// 依赖：
//   Runtime/backend/ocl/ocl_buffer.h。

#include "Runtime/backend/ocl/ocl_buffer.h"

namespace mobilenet {
namespace ocl {

OclBuffer::OclBuffer(cl_mem mem, U64 bytes) : mem_(mem), bytes_(bytes) {}

void* OclBuffer::DevicePtr() const {
  return reinterpret_cast<void*>(mem_);
}

U64 OclBuffer::Bytes() const { return bytes_; }

}  // namespace ocl
}  // namespace mobilenet
