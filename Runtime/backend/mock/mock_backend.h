// 模块：Mock 后端
//
// 职责：
//   用主机内存与 CPU 参考 Kernel 实现 Backend 接口，供主机侧契约测试、
//   单算子精度测试与插件机制验证使用。不追求性能，单线程确定性。
//
// 边界：
//   - 不是产品后端；不作为 OCL 失败后的静默回退；
//   - 只实现 Qwen3 所需的 8 个融合算子与 Input/Output；
//   - 不做设备同步、真实异步传输或性能优化；
//   - 不拥有 Graph/OpEnv，不解析模型包。
//
// 依赖：
//   Runtime/backend/backend.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_BACKEND_MOCK_MOCK_BACKEND_H_
#define MOBILENET_RUNTIME_BACKEND_MOCK_MOCK_BACKEND_H_

#include <memory>

#include "Runtime/backend/backend.h"

namespace mobilenet {
namespace mock {

// 主机 CPU 参考后端。
class MockBackend final : public Backend {
 public:
  MockBackend();

  ~MockBackend() override;

  MockBackend(const MockBackend&) = delete;
  MockBackend& operator=(const MockBackend&) = delete;
  MockBackend(MockBackend&&) = delete;
  MockBackend& operator=(MockBackend&&) = delete;

  // 初始化后端。
  //
  // 参数：
  //   config：设备序号必须非负，队列数必须大于 0；
  //           其余选项忽略。
  // 返回：
  //   成功返回 kOk；重复 Init 返回 kStateError；非法配置返回
  //   kInvalidArgument。
  Status Init(const BackendConfig& config) override;

  // 释放后端资源；幂等。
  Status Deinit() override;

  // 返回能力声明；未初始化返回 kStateError。
  Status Capability(BackendCapability* capability) const override;

  // 分配主机内存模拟设备缓冲。
  //
  // 参数：
  //   desc：bytes 必须大于 0 且不超过 max_buffer_bytes；alignment 必须为
  //         2 的幂（0 表示默认 64）；
  //   buffer：输出参数。
  // 返回：
  //   成功返回 kOk；参数非法/超限返回 kBufferError；分配失败返回
  //   kDeviceOom。
  Status CreateBuffer(const BufferDesc& desc,
                      std::unique_ptr<Buffer>* buffer) override;

  // 释放缓冲；允许空指针。
  Status DestroyBuffer(std::unique_ptr<Buffer> buffer) override;

  // 权重转换/上传。
  //
  // 参数：
  //   desc：host_ptr/bytes 必须有效；source_dtype/target_dtype 必须受支持；
  //   buffer：输出常驻缓冲。
  // 返回：
  //   成功返回 kOk；参数非法返回 kInvalidArgument；不支持 dtype 返回
  //   kUnsupportedDtype；分配失败返回 kDeviceOom。
  Status ConvertParam(const ParamDesc& desc,
                      std::unique_ptr<Buffer>* buffer) override;

  // 主机到设备拷贝；Mock 实现为 memcpy。
  Status Upload(void* device_ptr, const void* host_ptr, uint64_t bytes,
                Stream* stream) override;

  // 设备到主机拷贝；Mock 实现为 memcpy。
  Status Download(void* host_ptr, const void* device_ptr, uint64_t bytes,
                  Stream* stream) override;

  // 按 kernel_key 创建 CPU 参考 Kernel。
  //
  // 参数：
  //   desc：kernel_key 必须是 8 个已支持 key 之一；
  //   kernel：输出参数。
  // 返回：
  //   成功返回 kOk；未知 key 返回 kUnsupportedOp；其它失败返回
  //   kKernelError。
  Status CreateKernel(const KernelDesc& desc,
                      std::unique_ptr<Kernel>* kernel) override;

  // 释放 Kernel；允许空指针。
  Status DestroyKernel(std::unique_ptr<Kernel> kernel) override;

  // 同步；Mock 无在途任务，直接返回 kOk；未初始化返回 kStateError。
  Status Synchronize(Stream* stream) override;

 private:
  // 是否已完成 Init。
  bool initialized_ = false;

  // 能力声明。
  BackendCapability capability_;
};

// 创建 Mock 后端。
//
// 参数：
//   config：传给 MockBackend::Init 的配置；工厂本身不调用 Init；
//   backend：输出参数，不允许为空。
// 返回：
//   成功返回 kOk 并写入 MockBackend；否则返回 kInvalidArgument。
Status CreateMockBackend(const BackendConfig& config,
                         std::unique_ptr<Backend>* backend);

}  // namespace mock
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_MOCK_MOCK_BACKEND_H_
