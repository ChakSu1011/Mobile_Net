// 模块：Runtime 输入输出绑定接口
//
// 职责：
//   定义 Executor 向 Engine 绑定图输入/输出主机缓冲的统一接口；Engine 实现
//   该接口并把主机 Storage 写入 Input/Output op 的 attr["storage"]。
//
// 边界：
//   - 不负责设备缓冲分配或 H2D/D2H；
//   - 不暴露 Graph 的 Op/Tensor 访问器；
//   - 绑定只在单步内有效，由 Executor 负责建立与失效。
//
// 依赖：
//   Runtime/common/status.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_COMMON_IO_BINDER_H_
#define MOBILENET_RUNTIME_COMMON_IO_BINDER_H_

#include <string>

#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {

namespace runtime_graph {
struct Storage;
}  // namespace runtime_graph

namespace runtime {

// 输入输出绑定接口。
//
// 功能：
//   Executor 在执行前调用 SetInput/SetOutput，把本步主机缓冲描述交给
//   Engine；Engine 校验名称、形状、精度与字节数后写入对应 op 的 storage。
//
// 边界：
//   - 只允许推理线程调用；
//   - host_storage.data_ptr 必须非空且 device_type 为 kHost；
//   - 返回失败时 Executor 必须终止本步。
class IoBinder {
 public:
  virtual ~IoBinder() = default;

  // 绑定一个图输入的主机缓冲。
  //
  // 参数：
  //   name：Input op 名称（manifest inputs 中的名称）；
  //   host_storage：主机缓冲描述，包含 shape/precision/layout/byte_size/
  //                 data_ptr，device_type 必须为 kHost。
  // 返回：
  //   成功返回 kOk；名称未注册、类型不匹配或数据指针为空返回
  //   kInvalidArgument。
  virtual Status SetInput(const std::string& name,
                          const runtime_graph::Storage& host_storage) = 0;

  // 绑定一个图输出的主机缓冲。
  //
  // 参数：
  //   name：Output op 名称（manifest outputs 中的名称）；
  //   host_storage：主机缓冲描述，byte_size 必须与源张量一致。
  // 返回：
  //   成功返回 kOk；名称未注册、尺寸不匹配或数据指针为空返回
  //   kInvalidArgument。
  virtual Status SetOutput(const std::string& name,
                           const runtime_graph::Storage& host_storage) = 0;
};

}  // namespace runtime
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_COMMON_IO_BINDER_H_
