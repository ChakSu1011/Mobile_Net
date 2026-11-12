// 模块：Runtime Graph Storage 实现
//
// 职责：
//   实现 Storage 的稳定文本表示，供 dump_ir、日志和测试使用。
//
// 边界：
//   只格式化元数据，不读取 data_ptr 指向的内存，不输出真实地址。
//
// 依赖：
//   runtime_graph/storage.h。

#include "runtime_graph/storage.h"

#include <sstream>
#include <string>

namespace mobilenet {
namespace runtime_graph {
namespace {

// 返回 Shape 逻辑维度的文本表示。
//
// 参数：
//   shape：待格式化的 shape。
// 返回：
//   形如 "[1,8,2048]" 的字符串；max_dim 越界时返回 "invalid"。
std::string ShapeToString(const Shape& shape) {
  if (shape.max_dim < 0 || shape.max_dim > kMaxShapeDim) {
    return "invalid";
  }
  std::ostringstream stream;
  stream << "[";
  for (int64_t index = 0; index < shape.max_dim; ++index) {
    if (index != 0) {
      stream << ",";
    }
    stream << shape.dims[static_cast<size_t>(index)];
  }
  stream << "]";
  return stream.str();
}

}  // namespace

std::string StorageToString(const Storage& storage) {
  std::ostringstream stream;
  stream << "storage(precision=" << PrecisionName(storage.precision)
         << ",layout=" << LayoutName(storage.layout)
         << ",device=" << DeviceTypeName(storage.device_type)
         << ",shape=" << ShapeToString(storage.shape)
         << ",offset=" << storage.offset
         << ",bytes=" << storage.byte_size
         << ",data=" << (storage.data_ptr == nullptr ? "unset" : "set")
         << ",reusable=" << (storage.reusable ? "true" : "false") << ")";
  return stream.str();
}

}  // namespace runtime_graph
}  // namespace mobilenet
