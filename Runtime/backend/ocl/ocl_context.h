// 模块：OCL 设备与上下文
//
// 职责：
//   加载 OpenCL 运行库，选择 GPU 设备，创建 Context/CommandQueue，并查询
//   设备能力；为 OclBackend/OclProgramCache/Kernel 提供统一句柄。
//
// 边界：
//   - 不创建 Buffer/Kernel；不执行传输；
//   - 不暴露到 Backend 公共头；CL 类型只在 ocl/ 内部；
//   - 单设备、单队列；不要求线程安全。
//
// 依赖：
//   Runtime/backend/backend.h（BackendConfig）、ocl_api.h、ocl_utils.h、
//   Runtime/common/status.h、Utils/cpp/types.h。

#ifndef MOBILENET_RUNTIME_BACKEND_OCL_OCL_CONTEXT_H_
#define MOBILENET_RUNTIME_BACKEND_OCL_OCL_CONTEXT_H_

#include <array>
#include <string>

#include "Runtime/backend/backend.h"
#include "Runtime/backend/ocl/ocl_api.h"
#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace ocl {

// OCL 设备能力快照。
struct OclDeviceInfo {
  // 平台信息。
  std::string platform_name;
  std::string platform_vendor;
  std::string platform_version;

  // 设备信息。
  std::string device_name;
  std::string device_vendor;
  std::string device_version;
  std::string driver_version;
  std::string opencl_c_version;
  std::string extensions;

  // 计算资源。
  U32 compute_units = 0;
  U32 max_clock_mhz = 0;
  Usize max_work_group_size = 0;
  U32 max_work_item_dimensions = 0;
  std::array<Usize, 3> max_work_item_sizes{{0, 0, 0}};
  U32 preferred_vector_width_half = 0;
  U32 preferred_vector_width_float = 0;

  // 内存。
  U64 global_mem_bytes = 0;
  U64 max_mem_alloc_bytes = 0;
  U64 local_mem_bytes = 0;
  U32 local_mem_type = 0;
  U64 global_cache_bytes = 0;
  U32 cacheline_bytes = 0;
  U32 mem_base_addr_align_bits = 0;
  U64 max_constant_buffer_bytes = 0;
  U32 max_constant_args = 0;
  U64 svm_capabilities = 0;

  // 功能。
  Bool image_support = false;
  Bool fp16 = false;
  Bool subgroups = false;
  U64 profiling_timer_resolution_ns = 0;
};

// OCL 设备/上下文/队列。
class OclContext {
 public:
  OclContext() = default;
  ~OclContext();

  OclContext(const OclContext&) = delete;
  OclContext& operator=(const OclContext&) = delete;
  OclContext(OclContext&&) = delete;
  OclContext& operator=(OclContext&&) = delete;

  // 加载库、选择 GPU、创建 Context 与 Queue。
  //
  // 参数：
  //   config：device_id 为 GPU 设备索引；options["opencl_library"] 可指定库路径。
  // 返回：
  //   成功返回 kOk；无库/无 GPU/创建失败返回 kBackendInitError/kDeviceError。
  Status Init(const BackendConfig& config);

  // 释放 Queue/Context 与库句柄；幂等。
  Status Deinit();

  // 是否已初始化。
  Bool initialized() const { return initialized_; }

  // 返回动态加载的 API。
  OclApi* api() { return &api_; }

  // 返回函数表；未初始化时可能为空。
  OclFunctions* functions() { return api_.functions(); }

  // 返回 Context。
  cl_context context() const { return context_; }

  // 返回设备。
  cl_device_id device() const { return device_; }

  // 返回命令队列。
  cl_command_queue queue() const { return queue_; }

  // 返回设备能力快照。
  const OclDeviceInfo& info() const { return info_; }

 private:
  // 选择平台与 GPU 设备。
  Status SelectDevice(const std::string& library_path, I32 device_id);

  // 查询平台/设备信息。
  Status QueryInfo();

  // 动态 API。
  OclApi api_;

  // 选择结果。
  cl_platform_id platform_ = nullptr;
  cl_device_id device_ = nullptr;
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;

  // 设备信息。
  OclDeviceInfo info_;

  // 是否初始化。
  Bool initialized_ = false;
};

}  // namespace ocl
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OCL_OCL_CONTEXT_H_
