// 模块：Session JSON 序列化
//
// 职责：
//   提供轻量流式 JSON 构建器，供 OpenAI 响应、错误体、健康检查等使用。
//
// 边界：
//   - 只做序列化，不解析 JSON；
//   - 字符串按 RFC 8259 转义；非 ASCII UTF-8 原样输出；
//   - 结构使用错误通过 failed() 返回，不抛出异常；
//   - 浮点非有限值输出 null。
//
// 依赖：
//   C++17 标准库、Utils/cpp/types.h。

#ifndef MOBILENET_SERVICE_SESSION_JSON_WRITER_H_
#define MOBILENET_SERVICE_SESSION_JSON_WRITER_H_

#include <string>
#include <vector>

#include "Utils/cpp/types.h"

namespace mobilenet {
namespace service {

// JSON 字符串转义。
//
// 参数：
//   text：UTF-8 文本；
//   escaped：输出参数，非空；写入不含首尾引号的转义内容。
void EscapeJsonString(const std::string& text, std::string* escaped);

// 流式 JSON 构建器。
//
// 用法：
//   JsonWriter writer;
//   writer.BeginObject();
//   writer.Key("id"); writer.String("x");
//   writer.EndObject();
//   std::string json = writer.str();
//
// 边界：
//   - 结构错误（例如对象中缺少 Key）置 failed() 为 true；
//   - 不自动添加换行。
class JsonWriter {
 public:
  JsonWriter();

  // 开始对象/数组。
  void BeginObject();
  void BeginArray();

  // 结束对象/数组。
  void EndObject();
  void EndArray();

  // 写入对象键；随后必须写一个值。
  void Key(const std::string& key);

  // 写入值。
  void String(const std::string& value);
  void Int(I64 value);
  void Double(FP64 value);
  void Boolean(Bool value);
  void Null();

  // 直接写入已序列化的 JSON 值。
  void Raw(const std::string& json);

  // 返回序列化结果。
  const std::string& str() const { return out_; }

  // 返回是否发生结构错误。
  Bool failed() const { return failed_; }

 private:
  // 当前容器状态。
  struct Level {
    // 是否为对象。
    Bool is_object = false;

    // 是否尚未写入第一个成员/元素。
    Bool first = true;
  };

  // 写值前处理逗号与对象键状态。
  void BeforeValue();

  // 写完值后更新对象键状态。
  void AfterValue();

  // 输出字符串（含引号）并转义。
  void WriteString(const std::string& value);

  // 序列化结果。
  std::string out_;

  // 容器栈。
  std::vector<Level> levels_;

  // 对象中是否正在等待键。
  Bool expect_key_ = false;

  // 是否发生结构错误。
  Bool failed_ = false;
};

}  // namespace service
}  // namespace mobilenet

#endif  // MOBILENET_SERVICE_SESSION_JSON_WRITER_H_
