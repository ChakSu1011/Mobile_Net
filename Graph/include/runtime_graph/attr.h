// 模块：Runtime Graph Op 属性
//
// 职责：
//   定义 Op 的静态属性容器 AttrMap 与属性值 AttrValue，并提供稳定的文本
//   格式化接口；Input/Output op 通过 attr["storage"] 承载构图期或运行期
//   绑定的 Storage。
//
// 边界：
//   - 只支持标量、字符串、一维数组和 Storage 这几类值，不保存任意对象；
//   - 不负责属性语义校验，具体 op 在 OpEnv 初始化或 Finalizer 中解释；
//   - 不保存权重数据指针以外的运行时资源，权重地址由 constant tensor 承载。
//
// 依赖：
//   runtime_graph/storage.h、C++17 标准库。

#ifndef MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_ATTR_H_
#define MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_ATTR_H_

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include "runtime_graph/storage.h"

namespace mobilenet {
namespace runtime_graph {

// Op 属性值。
//
// 允许的类型：
//   bool、int64_t、double、std::string、std::vector<int64_t>、
//   std::vector<double>、std::vector<std::string>、Storage。
using AttrValue =
    std::variant<bool, int64_t, double, std::string, std::vector<int64_t>,
                 std::vector<double>, std::vector<std::string>, Storage>;

// Op 属性表，按属性名排序，保证 dump_ir 输出稳定。
using AttrMap = std::map<std::string, AttrValue>;

// 返回属性值的稳定文本表示。
//
// 参数：
//   value：属性值。
// 返回：
//   标量输出数字或布尔文本；字符串输出原文；数组输出 "[a,b]"；
//   Storage 输出 StorageToString() 的结果。
std::string AttrValueToString(const AttrValue& value);

}  // namespace runtime_graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_INCLUDE_RUNTIME_GRAPH_ATTR_H_
