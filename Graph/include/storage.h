// 模块：Runtime Graph Storage 结构
//
// 职责：
//   描述 tensor、constant 或 tempdata 的物理存储信息，包括 shape、精度、
//   布局、设备类别、workspace offset、字节数、数据地址和复用标记；Storage
//   是 Finalizer、Manager 与 OpEnv 之间的共同载体。
//
// 边界：
//   - 只保存数据描述，不拥有内存，不负责分配/释放 data_ptr；
//   - 不定义 shape / precision / layout 推导策略；
//   - 不携带具体 GPU/OCL 设备号或 kernel 句柄；
//   - data_ptr 的语义由 device_type 决定，调用方必须先检查设备类别。
//
// 依赖：
//   shape.h、types.h。

#ifndef MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_STORAGE_H_
#define MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_STORAGE_H_

#include <cstddef>
#include <string>

#include "shape.h"
#include "types.h"

namespace mobilenet {
namespace runtime_graph {

// Storage 描述 tensor 或 constant 的物理存储信息。
//
// 功能：
//   作为推导结果、显存规划结果和底层算子调用参数之间的共同数据结构。
//
// 边界：
//   - 默认 reusable 为 true，constant tensor 在 Graph::CreateConstant 中
//     会被强制置为 false；
//   - offset 只对 workspace buffer 有意义，绝对值地址由 data_ptr 表达；
//   - byte_size 由调用方按权重加载或显存规划结果填写，Storage 不自动推导。
struct Storage {
  // tensor 维度。
  Shape shape;

  // 数据精度。
  Precision precision = Precision::kUnknown;

  // 数据布局。
  Layout layout = Layout::kRowMajor;

  // 当前 data_ptr 所在设备类别。
  DeviceType device_type = DeviceType::kUnknown;

  // workspace buffer 中的逻辑偏移，单位由 Manager 约定（当前为字节）。
  size_t offset = 0;

  // 当前 storage 对应的字节数。
  size_t byte_size = 0;

  // 实际数据地址；不拥有该地址指向的内存。
  void* data_ptr = nullptr;

  // 是否允许参与 workspace 复用。
  bool reusable = true;
};

// 返回 Storage 的稳定文本表示，用于 dump_ir 和日志。
//
// 参数：
//   storage：待格式化的 storage。
// 返回：
//   形如 "storage(precision=float16,layout=row_major,device=host,
//   shape=[1,8],offset=0,bytes=16,data=set,reusable=true)" 的字符串；
//   不输出真实指针值，避免测试结果随进程地址变化。
std::string StorageToString(const Storage& storage);

}  // namespace runtime_graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_STORAGE_H_
