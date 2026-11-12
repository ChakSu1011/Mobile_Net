// 模块：Runtime Graph Op 属性实现
//
// 职责：
//   实现 AttrValueToString()，把 variant 属性值格式化为稳定文本。
//
// 边界：
//   只做格式化，不解释属性语义，不读取 Storage 指向的数据。
//
// 依赖：
//   runtime_graph/attr.h、runtime_graph/storage.h。

#include "runtime_graph/attr.h"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "runtime_graph/storage.h"

namespace mobilenet {
namespace runtime_graph {
namespace {

// 把多个可调用对象聚合成一个 visitor，供 std::visit 使用。
template <typename... Callables>
struct Overloaded : Callables... {
  using Callables::operator()...;
};

// 推导 Overloaded 的模板参数。
template <typename... Callables>
Overloaded(Callables...) -> Overloaded<Callables...>;

// 把数值数组格式化为 "[a,b]"。
//
// 参数：
//   values：数值数组。
// 返回：
//   逗号分隔的数组文本。
template <typename T>
std::string JoinNumbers(const std::vector<T>& values) {
  std::ostringstream stream;
  stream << "[";
  for (size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      stream << ",";
    }
    stream << values[index];
  }
  stream << "]";
  return stream.str();
}

// 把字符串数组格式化为 "[a,b]"。
//
// 参数：
//   values：字符串数组。
// 返回：
//   逗号分隔的数组文本；不额外加引号，保持 dump_ir 紧凑。
std::string JoinStrings(const std::vector<std::string>& values) {
  std::ostringstream stream;
  stream << "[";
  for (size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      stream << ",";
    }
    stream << values[index];
  }
  stream << "]";
  return stream.str();
}

}  // namespace

std::string AttrValueToString(const AttrValue& value) {
  return std::visit(
      Overloaded{
          [](bool item) { return std::string(item ? "true" : "false"); },
          [](int64_t item) { return std::to_string(item); },
          [](double item) {
            std::ostringstream stream;
            stream << item;
            return stream.str();
          },
          [](const std::string& item) { return item; },
          [](const std::vector<int64_t>& item) { return JoinNumbers(item); },
          [](const std::vector<double>& item) { return JoinNumbers(item); },
          [](const std::vector<std::string>& item) {
            return JoinStrings(item);
          },
          [](const Storage& item) { return StorageToString(item); },
      },
      value);
}

}  // namespace runtime_graph
}  // namespace mobilenet
