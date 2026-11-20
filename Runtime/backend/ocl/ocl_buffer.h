// 模块：OCL 设备缓冲
//
// 职责：
//   把 cl_mem 封装为 Runtime Buffer；DevicePtr() 返回 cl_mem 的不透明
//   void* 表示，供 Backend::Upload/Download 使用；Kernel 通过 mem() 取句柄。
//
// 边界：
//   - 不拥有/释放 cl_mem；释放由 OclBackend::DestroyBuffer 负责；
//   - 不记录 offset；Arena 子偏移由 TensorView.offset 表达；
//   - 不初始化内容。
//
// 依赖：
//   Runtime/backend/backend.h、ocl_api.h、Utils/cpp/types.h。

#ifndef MOBILENET_RUNTIME_BACKEND_OCL_OCL_BUFFER_H_
#define MOBILENET_RUNTIME_BACKEND_OCL_OCL_BUFFER_H_

#include "Runtime/backend/backend.h"
#include "Runtime/backend/ocl/ocl_api.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace ocl {

// OCL 设备缓冲。
class OclBuffer final : public Buffer {
 public:
  // 创建缓冲包装。
  //
  // 参数：
  //   mem：已创建的 cl_mem，非空；
  //   bytes：字节数。
  OclBuffer(cl_mem mem, U64 bytes);

  ~OclBuffer() override = default;

  OclBuffer(const OclBuffer&) = delete;
  OclBuffer& operator=(const OclBuffer&) = delete;
  OclBuffer(OclBuffer&&) = delete;
  OclBuffer& operator=(OclBuffer&&) = delete;

  // 返回 cl_mem 的 void* 表示。
  void* DevicePtr() const override;

  // 返回字节数。
  U64 Bytes() const override;

  // 返回 cl_mem 句柄。
  cl_mem mem() const { return mem_; }

 private:
  // cl_mem 句柄；不拥有。
  cl_mem mem_ = nullptr;

  // 字节数。
  U64 bytes_ = 0;
};

}  // namespace ocl
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OCL_OCL_BUFFER_H_
