// 模块：Runtime Graph Shape 结构
//
// 职责：
//   定义 tensor 的逻辑维度 dims 与物理 pitch，并提供逻辑元素数量、物理
//   元素数量的计算接口；Shape 是 Storage 的内嵌成员。
//
// 边界：
//   - 只保存定长维度数组，不负责动态扩维、符号维或 shape 推导；
//   - 不保存 dtype、layout 或 data pointer；
//   - 维度值使用 int64_t，负数视为非法输入。
//
// 依赖：
//   types.h（kMaxShapeDim）。

#ifndef MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_SHAPE_H_
#define MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_SHAPE_H_

#include <array>
#include <cstdint>

#include "types.h"

namespace mobilenet {
namespace runtime_graph {

// Shape 描述 tensor 的逻辑维度和物理 pitch。
//
// 功能：
//   保存最多 kMaxShapeDim 个逻辑维度与对应的物理维度，支持计算逻辑元素
//   数量和物理 pitch 元素数量。
//
// 边界：
//   - 默认构造后 max_dim 为 0，dims/pitch 的每个槽位为 1；
//   - 不校验 pitch 与 dims 的语义关系，只做乘积计算；
//   - 不负责分配内存，也不携带 stride 的字节单位。
struct Shape {
  // 构造空 shape：所有槽位置 1，max_dim 置 0。
  Shape();

  // 返回逻辑 shape 的元素数量。
  //
  // 返回：
  //   dims[0, max_dim) 的乘积；max_dim 越界、维度为负或乘积溢出时返回 -1。
  int64_t GetShapeSize() const;

  // 返回物理 pitch 对应的元素数量。
  //
  // 返回：
  //   pitch[0, max_dim) 的乘积；max_dim 越界、维度为负或乘积溢出时返回 -1。
  int64_t GetByteSize() const;

  // 逻辑维度数组，默认每一维为 1。
  std::array<int64_t, kMaxShapeDim> dims;

  // 物理维度数组，默认每一维为 1。
  std::array<int64_t, kMaxShapeDim> pitch;

  // 实际有效维度数量，取值范围 [0, kMaxShapeDim]。
  int64_t max_dim;
};

}  // namespace runtime_graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_SHAPE_H_
