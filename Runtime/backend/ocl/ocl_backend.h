// 模块：OCL Backend
//
// 职责：
//   实现 Runtime Backend 接口：设备/上下文/队列、Buffer、H2D/D2H、
//   权重转换上传、Program 缓存、同步；注册名 "ocl"。
//
// 边界：
//   - 不实现 Kernel（P4.2 起逐算子接入）；
//   - 不解析模型包、不调度、不管理 KV 策略；
//   - 公共头不出现 CL 类型；CL 类型只在 ocl/ 内部。
//
// 依赖：
//   Runtime/backend/backend.h、ocl_context.h、ocl_program_cache.h、
//   Runtime/common/status.h、Utils/cpp/types.h。

#ifndef MOBILENET_RUNTIME_BACKEND_OCL_OCL_BACKEND_H_
#define MOBILENET_RUNTIME_BACKEND_OCL_OCL_BACKEND_H_

#include <memory>

#include "Runtime/backend/backend.h"
#include "Runtime/backend/ocl/ocl_context.h"
#include "Runtime/backend/ocl/ocl_program_cache.h"
#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace ocl {

// OCL 后端。
class OclBackend final : public Backend {
 public:
  OclBackend() = default;
  ~OclBackend() override;

  OclBackend(const OclBackend&) = delete;
  OclBackend& operator=(const OclBackend&) = delete;
  OclBackend(OclBackend&&) = delete;
  OclBackend& operator=(OclBackend&&) = delete;

  // 初始化 OpenCL 设备/上下文/队列与 Program 缓存。
  //
  // 参数：
  //   config：device_id 为 GPU 索引；options 可含 opencl_library/cache_dir。
  // 返回：
  //   成功返回 kOk；无库/无 GPU/初始化失败返回 kBackendInitError。
  Status Init(const BackendConfig& config) override;

  // 释放 Program 缓存、Context/Queue；幂等。
  Status Deinit() override;

  // 返回能力声明；未初始化返回 kStateError。
  Status Capability(BackendCapability* capability) const override;

  // 创建 cl_mem 缓冲。
  Status CreateBuffer(const BufferDesc& desc,
                      std::unique_ptr<Buffer>* buffer) override;

  // 释放缓冲；允许空指针。
  Status DestroyBuffer(std::unique_ptr<Buffer> buffer) override;

  // 权重转换/上传：FP16 直通；FP32/BF16→FP16。
  Status ConvertParam(const ParamDesc& desc,
                      std::unique_ptr<Buffer>* buffer) override;

  // 主机到设备；device_ptr 为 OclBuffer::DevicePtr()。
  Status Upload(void* device_ptr, const void* host_ptr, U64 bytes,
                Stream* stream) override;

  // 设备到主机。
  Status Download(void* host_ptr, const void* device_ptr, U64 bytes,
                  Stream* stream) override;

  // P4.1 暂无 Kernel；返回 kUnsupportedOp。P4.2 接入 8 个算子。
  Status CreateKernel(const KernelDesc& desc,
                      std::unique_ptr<Kernel>* kernel) override;

  // 释放 Kernel；允许空指针。
  Status DestroyKernel(std::unique_ptr<Kernel> kernel) override;

  // clFinish；每 Step 一次。
  Status Synchronize(Stream* stream) override;

 private:
  // 非拥有上下文。
  OclContext context_;

  // Program 缓存。
  OclProgramCache programs_;

  // 能力声明。
  BackendCapability capability_;

  // 是否初始化。
  Bool initialized_ = false;
};

// 创建 OCL 后端。
//
// 参数：
//   config：工厂不调用 Init；
//   backend：输出参数，不允许为空。
// 返回：
//   成功返回 kOk 并写入 OclBackend。
Status CreateOclBackend(const BackendConfig& config,
                        std::unique_ptr<Backend>* backend);

}  // namespace ocl
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OCL_OCL_BACKEND_H_
