// 模块：Backend 公开接口
//
// 职责：
//   定义 Runtime 后端必须实现的最小能力集合：主机/设备拷贝、Kernel 创建与
//   销毁、流同步。Graph/OpEnv 只依赖本接口，不感知 CUDA/OpenCL/Vulkan 细节。
//
// 边界：
//   - 不定义 MemoryPlan、权重加载或模型语义；
//   - 不拥有 Graph 的 arena 内存；arena 由 Loader/Destroy 管理。
//
// 依赖：
//   Runtime/common/status.h、Runtime/kernel/kernel.h。

#ifndef MOBILENET_RUNTIME_BACKEND_BACKEND_H_
#define MOBILENET_RUNTIME_BACKEND_BACKEND_H_

#include <cstdint>

#include "Runtime/common/status.h"
#include "Runtime/kernel/kernel.h"

namespace mobilenet {

// 后端执行流。
//
// 功能：
//   表示异步拷贝/计算使用的流；具体实现由后端定义。
//
// 边界：
//   基类不包含任何状态；Graph/OpEnv 只持有指针。
class Stream {
 public:
  virtual ~Stream() = default;
};

// Runtime 后端抽象接口。
//
// 功能：
//   提供 Graph/OpEnv 所需的最小设备能力。
//
// 边界：
//   - 不负责图构建、校验或拓扑排序；
//   - CreateKernel 失败时必须保证不泄漏部分构造的 Kernel。
class Backend {
 public:
  virtual ~Backend() = default;

  // 主机到设备的同步/异步拷贝。
  //
  // 参数：
  //   device_ptr：设备目标地址，非空；
  //   host_ptr：主机源地址，非空；
  //   bytes：拷贝字节数；
  //   stream：执行流，可为空表示默认流。
  // 返回：
  //   Status::Ok() 或 BUFFER_ERROR/DEVICE_ERROR。
  virtual Status Upload(void* device_ptr, const void* host_ptr, uint64_t bytes,
                        Stream* stream) = 0;

  // 设备到主机的同步/异步拷贝。
  //
  // 参数：
  //   host_ptr：主机目标地址，非空；
  //   device_ptr：设备源地址，非空；
  //   bytes：拷贝字节数；
  //   stream：执行流，可为空表示默认流。
  // 返回：
  //   Status::Ok() 或 BUFFER_ERROR/DEVICE_ERROR。
  virtual Status Download(void* host_ptr, const void* device_ptr,
                          uint64_t bytes, Stream* stream) = 0;

  // 创建 Kernel。
  //
  // 参数：
  //   desc：Kernel 描述；
  //   kernel：输出参数，成功时写入非空指针，失败时保持未修改。
  // 返回：
  //   Status::Ok()、UNSUPPORTED_OP 或 KERNEL_ERROR。
  virtual Status CreateKernel(const KernelDesc& desc, Kernel** kernel) = 0;

  // 销毁 Kernel。
  //
  // 参数：
  //   kernel：由 CreateKernel 返回的指针；允许为空。
  // 返回：
  //   Status::Ok() 或 KERNEL_ERROR。
  virtual Status DestroyKernel(Kernel* kernel) = 0;

  // 同步执行流。
  //
  // 参数：
  //   stream：执行流，可为空表示默认流。
  // 返回：
  //   Status::Ok() 或 DEVICE_ERROR。
  virtual Status Synchronize(Stream* stream) = 0;
};

}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_BACKEND_H_
