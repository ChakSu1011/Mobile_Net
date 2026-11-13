// 模块：Runtime Graph Shape 实现
//
// 职责：
//   实现 Shape 默认构造、逻辑元素数量与物理 pitch 元素数量计算。
//
// 边界：
//   只做维度乘积与合法性检查，不分配内存、不推导 shape。
//
// 依赖：
//   runtime_graph/shape.h。

#include "runtime_graph/shape.h"

#include <cstddef>
#include <limits>

namespace mobilenet {
namespace runtime_graph {
namespace {

// 计算 dims[0, count) 的乘积。
//
// 参数：
//   values：维度数组；
//   count：有效维度数量，调用方保证在 [0, kMaxShapeDim] 内。
// 返回：
//   乘积；遇到负维度或 int64_t 溢出时返回 -1。
int64_t MultiplyDims(const std::array<int64_t, kMaxShapeDim>& values,
                     int64_t count) {
  int64_t product = 1;
  for (int64_t index = 0; index < count; ++index) {
    const int64_t value = values[static_cast<size_t>(index)];
    if (value < 0) {
      return -1;
    }
    if (value != 0 && product > std::numeric_limits<int64_t>::max() / value) {
      return -1;
    }
    product *= value;
  }
  return product;
}

}  // namespace

Shape::Shape() : max_dim(0) {
  dims.fill(1);
  pitch.fill(1);
}

int64_t Shape::GetShapeSize() const {
  if (max_dim < 0 || max_dim > kMaxShapeDim) {
    return -1;
  }
  return MultiplyDims(dims, max_dim);
}

int64_t Shape::GetByteSize() const {
  if (max_dim < 0 || max_dim > kMaxShapeDim) {
    return -1;
  }
  return MultiplyDims(pitch, max_dim);
}

}  // namespace runtime_graph
}  // namespace mobilenet
