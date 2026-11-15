// 模块：Backend 公开接口
//
// 职责：
//   定义 Runtime 后端必须实现的统一能力：初始化/释放、能力查询、设备缓冲
//   创建/销毁、权重转换上传、主机-设备传输、Kernel 创建/销毁与流同步。
//   Loader/Executor/OpEnv 只依赖本接口，不感知 CUDA/OpenCL/Vulkan 细节。
//
// 边界：
//   - 不定义 Graph、调度、KV 策略或模型语义；
//   - 不拥有 Graph 的 arena 内存；缓冲句柄由 Runtime 创建并释放；
//   - 公共头文件不得出现厂商类型；设备句柄均为不透明类型；
//   - 由 Engine 串行调用，不要求内部线程安全。
//
// 依赖：
//   Runtime/common/status.h、Runtime/common/data_types.h、
//   Runtime/backend/kernel.h、Graph/include/op_type.h。

#ifndef MOBILENET_RUNTIME_BACKEND_BACKEND_H_
#define MOBILENET_RUNTIME_BACKEND_BACKEND_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Graph/include/op_type.h"
#include "Runtime/backend/kernel.h"
#include "Runtime/common/data_types.h"
#include "Runtime/common/status.h"

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

// 设备缓冲句柄。
//
// 功能：
//   封装后端分配的设备内存，提供裸地址与字节数供 Kernel 使用。
//
// 边界：
//   - 不暴露厂商句柄；
//   - 生命周期由创建者（Runtime）通过 Backend::DestroyBuffer 管理；
//   - 不负责内容初始化。
class Buffer {
 public:
  virtual ~Buffer() = default;

  // 返回设备地址；未分配成功时不得返回空。
  virtual void* DevicePtr() const = 0;

  // 返回缓冲字节数。
  virtual uint64_t Bytes() const = 0;
};

// 缓冲用途。
//
// 说明：
//   用于后端选择内存类型/对齐策略；语义由 Runtime 约定。
enum class BufferUsage {
  kPersistent = 0,
  kActivation = 1,
  kIo = 2,
  kWorkspace = 3,
  kKv = 4,
};

// KV 物理布局。
//
// 说明：
//   kNhd：token-major [block_size, kv_heads, head_dim]；
//   kHnd：head-major，供特定后端 Kernel 使用。
enum class KvLayout {
  kNhd = 0,
  kHnd = 1,
};

// 缓冲分配描述。
//
// 字段：
//   bytes：需要的字节数；
//   alignment：起始地址对齐要求（字节）；
//   usage：用途，供后端选择内存类型。
struct BufferDesc {
  uint64_t bytes = 0;
  uint64_t alignment = 0;
  BufferUsage usage = BufferUsage::kPersistent;
};

// 后端初始化配置。
//
// 字段：
//   device_id：设备序号；
//   queue_count：队列数；
//   options：后端专有选项（不得包含影响上层的语义）。
struct BackendConfig {
  int32_t device_id = 0;
  uint32_t queue_count = 1;
  std::map<std::string, std::string> options;
};

// 权重转换/上传描述。
//
// 字段：
//   name：权重名（用于日志/缓存键）；
//   host_ptr：主机源地址，非空；
//   bytes：源字节数；
//   source_dtype：源精度；
//   target_dtype：目标精度。
struct ParamDesc {
  std::string name;
  const void* host_ptr = nullptr;
  uint64_t bytes = 0;
  DataType source_dtype = DataType::kUnknown;
  DataType target_dtype = DataType::kUnknown;
};

// 后端能力。
//
// 功能：
//   Loader/KVCache 在加载期据此校验精度、算子、传输方向、对齐与 KV 块大小。
//
// 边界：
//   - supported_ops 使用 runtime_graph::OpType；
//   - attention_block_size 为 KV 块大小的唯一来源（0 表示未声明）；
//   - 能力声明必须真实，禁止运行期静默降级。
struct BackendCapability {
  // 支持的精度集合。
  std::vector<DataType> supported_dtypes;

  // 支持的算子集合。
  std::vector<runtime_graph::OpType> supported_ops;

  // 是否支持主机到设备传输。
  bool h2d = false;

  // 是否支持设备到主机传输。
  bool d2h = false;

  // 单缓冲最大字节数。
  uint64_t max_buffer_bytes = 0;

  // 最小对齐要求。
  uint64_t min_alignment = 0;

  // Attention Kernel 的 KV 块大小（token 数）。
  int32_t attention_block_size = 0;

  // KV 物理布局。
  KvLayout kv_layout = KvLayout::kNhd;
};

// Runtime 后端抽象接口。
//
// 功能：
//   提供 Loader/Executor/OpEnv 所需的全部设备能力。
//
// 边界：
//   - 不负责图构建、校验、拓扑排序或调度；
//   - Init 全有或全无；Deinit 幂等；
//   - Create* 失败时不得泄漏部分构造的资源。
class Backend {
 public:
  virtual ~Backend() = default;

  // 初始化设备、上下文与队列。
  //
  // 参数：
  //   config：设备与队列配置。
  // 返回：
  //   成功返回 kOk；设备不可用返回 kBackendInitError/kDeviceError。
  virtual Status Init(const BackendConfig& config) = 0;

  // 释放后端资源。
  //
  // 返回：
  //   成功返回 kOk；幂等，可重复调用。
  virtual Status Deinit() = 0;

  // 查询后端能力。
  //
  // 参数：
  //   capability：输出参数，成功时写入能力集合。
  // 返回：
  //   成功返回 kOk；未初始化返回 kStateError。
  virtual Status Capability(BackendCapability* capability) const = 0;

  // 创建设备缓冲。
  //
  // 参数：
  //   desc：缓冲描述；
  //   buffer：输出参数，成功时写入非空句柄。
  // 返回：
  //   成功返回 kOk；超限/对齐不满足返回 kBufferError；OOM 返回 kDeviceOom。
  virtual Status CreateBuffer(const BufferDesc& desc,
                              std::unique_ptr<Buffer>* buffer) = 0;

  // 销毁设备缓冲。
  //
  // 参数：
  //   buffer：由 CreateBuffer 返回的句柄；允许为空。
  // 返回：
  //   成功返回 kOk；其它返回 kBufferError。
  virtual Status DestroyBuffer(std::unique_ptr<Buffer> buffer) = 0;

  // 转换并上传权重。
  //
  // 参数：
  //   desc：权重描述；
  //   buffer：输出参数，成功时写入常驻设备缓冲。
  // 返回：
  //   成功返回 kOk；精度不支持返回 kUnsupportedDtype；失败返回
  //   kBufferError/kDeviceError。
  virtual Status ConvertParam(const ParamDesc& desc,
                              std::unique_ptr<Buffer>* buffer) = 0;

  // 主机到设备拷贝。
  //
  // 参数：
  //   device_ptr：设备目标地址，非空；
  //   host_ptr：主机源地址，非空；
  //   bytes：拷贝字节数；
  //   stream：执行流，可为空表示默认流。
  // 返回：
  //   成功返回 kOk；失败返回 kBufferError/kDeviceError。
  virtual Status Upload(void* device_ptr, const void* host_ptr, uint64_t bytes,
                        Stream* stream) = 0;

  // 设备到主机拷贝。
  //
  // 参数：
  //   host_ptr：主机目标地址，非空；
  //   device_ptr：设备源地址，非空；
  //   bytes：拷贝字节数；
  //   stream：执行流，可为空表示默认流。
  // 返回：
  //   成功返回 kOk；失败返回 kBufferError/kDeviceError。
  virtual Status Download(void* host_ptr, const void* device_ptr,
                          uint64_t bytes, Stream* stream) = 0;

  // 创建 Kernel。
  //
  // 参数：
  //   desc：Kernel 描述；
  //   kernel：输出参数，成功时写入非空句柄。
  // 返回：
  //   成功返回 kOk；缺少算子返回 kUnsupportedOp；失败返回 kKernelError。
  virtual Status CreateKernel(const KernelDesc& desc,
                              std::unique_ptr<Kernel>* kernel) = 0;

  // 销毁 Kernel。
  //
  // 参数：
  //   kernel：由 CreateKernel 返回的句柄；允许为空。
  // 返回：
  //   成功返回 kOk；其它返回 kKernelError。
  virtual Status DestroyKernel(std::unique_ptr<Kernel> kernel) = 0;

  // 同步执行流。
  //
  // 参数：
  //   stream：执行流，可为空表示默认流。
  // 返回：
  //   成功返回 kOk；失败返回 kDeviceError。
  virtual Status Synchronize(Stream* stream) = 0;
};

}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_BACKEND_H_
