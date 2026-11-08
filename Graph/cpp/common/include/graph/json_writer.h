// 模块：确定性 JSON 写入器
//
// 职责：
//   为 Graph IR 与 RuntimeGraph 的 ToCanonicalJson() 提供无第三方依赖、
//   字段顺序可控、浮点格式稳定的 JSON 文本生成能力。
//
// 边界：
//   - 不解析 JSON；
//   - 不排序容器；调用方必须按设计规定的顺序写入；
//   - 输出无缩进、无空格，便于逐字节比较。
//
// 依赖：
//   仅依赖 C++17 标准库。

#ifndef MOBILENET_GRAPH_CPP_COMMON_INCLUDE_GRAPH_JSON_WRITER_H_
#define MOBILENET_GRAPH_CPP_COMMON_INCLUDE_GRAPH_JSON_WRITER_H_

#include <charconv>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mobilenet {
namespace graph {

// 确定性 JSON 写入器。
//
// 用法：
//   JsonWriter writer;
//   writer.BeginObject();
//   writer.Key("name");
//   writer.String("x");
//   writer.EndObject();
//   std::string json = writer.Take();
//
// 约束：
//   - Key() 只能在对象内调用；
//   - Begin/End 必须成对；
//   - 所有字符串按 JSON 规则转义，禁止直接拼接未转义内容。
class JsonWriter {
 public:
  JsonWriter() = default;

  // 开始对象。
  void BeginObject() {
    PrepareValue();
    output_.push_back('{');
    stack_.push_back(Frame{true, true});
  }

  // 结束对象。
  void EndObject() {
    output_.push_back('}');
    PopFrame();
  }

  // 开始数组。
  void BeginArray() {
    PrepareValue();
    output_.push_back('[');
    stack_.push_back(Frame{true, false});
  }

  // 结束数组。
  void EndArray() {
    output_.push_back(']');
    PopFrame();
  }

  // 写入对象键。
  //
  // 参数：
  //   key：键名；按 JSON 字符串转义。
  void Key(std::string_view key) {
    PrepareMember();
    WriteEscaped(key);
    output_.push_back(':');
    after_key_ = true;
  }

  // 写入字符串值。
  void String(std::string_view value) {
    PrepareValue();
    WriteEscaped(value);
  }

  // 写入有符号整数。
  void Int(int64_t value) {
    PrepareValue();
    char buffer[32];
    auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
    output_.append(buffer, result.ptr);
  }

  // 写入无符号整数。
  void UInt(uint64_t value) {
    PrepareValue();
    char buffer[32];
    auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
    output_.append(buffer, result.ptr);
  }

  // 写入浮点数。
  //
  // 使用最短可往返表示；NaN/Inf 写为 null，避免生成非法 JSON。
  void Double(double value) {
    PrepareValue();
    if (!std::isfinite(value)) {
      output_.append("null");
      return;
    }
    if (value == 0.0) {
      value = 0.0;  // 归一化 -0.0。
    }
    char buffer[64];
    auto result = std::to_chars(buffer, buffer + sizeof(buffer), value,
                                std::chars_format::general);
    output_.append(buffer, result.ptr);
  }

  // 写入布尔值。
  void Bool(bool value) {
    PrepareValue();
    output_.append(value ? "true" : "false");
  }

  // 写入 null。
  void Null() {
    PrepareValue();
    output_.append("null");
  }

  // 写入已格式化的 JSON 片段。
  //
  // 参数：
  //   raw：必须为合法 JSON；调用方负责正确性。
  void Raw(std::string_view raw) {
    PrepareValue();
    output_.append(raw);
  }

  // 取出结果并清空写入器。
  std::string Take() {
    std::string result = std::move(output_);
    output_.clear();
    stack_.clear();
    after_key_ = false;
    return result;
  }

 private:
  struct Frame {
    bool first = true;
    bool is_object = true;
  };

  // 准备写入一个值：若不是键后的值，则按需补逗号。
  void PrepareValue() {
    if (after_key_) {
      after_key_ = false;
      return;
    }
    if (!stack_.empty()) {
      if (!stack_.back().first) {
        output_.push_back(',');
      }
      stack_.back().first = false;
    }
  }

  // 准备写入对象成员键：按需补逗号。
  void PrepareMember() {
    if (!stack_.empty()) {
      if (!stack_.back().first) {
        output_.push_back(',');
      }
      stack_.back().first = false;
    }
  }

  // 结束当前容器。
  void PopFrame() {
    if (!stack_.empty()) {
      stack_.pop_back();
    }
  }

  // 写入 JSON 字符串转义形式。
  void WriteEscaped(std::string_view value) {
    static const char kHex[] = "0123456789abcdef";
    output_.push_back('"');
    for (unsigned char character : value) {
      switch (character) {
        case '"':
          output_.append("\\\"");
          break;
        case '\\':
          output_.append("\\\\");
          break;
        case '\b':
          output_.append("\\b");
          break;
        case '\f':
          output_.append("\\f");
          break;
        case '\n':
          output_.append("\\n");
          break;
        case '\r':
          output_.append("\\r");
          break;
        case '\t':
          output_.append("\\t");
          break;
        default:
          if (character < 0x20) {
            output_.append("\\u00");
            output_.push_back(kHex[(character >> 4) & 0x0F]);
            output_.push_back(kHex[character & 0x0F]);
          } else {
            output_.push_back(static_cast<char>(character));
          }
          break;
      }
    }
    output_.push_back('"');
  }

  std::string output_;
  std::vector<Frame> stack_;
  bool after_key_ = false;
};

}  // namespace graph
}  // namespace mobilenet

#endif  // MOBILENET_GRAPH_CPP_COMMON_INCLUDE_GRAPH_JSON_WRITER_H_
