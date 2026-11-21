// 模块：Session JSON 序列化实现
//
// 职责：
//   实现字符串转义与流式对象/数组构建。
//
// 边界：
//   不校验输入 UTF-8；调用方应保证字符串来自合法 JSON 或程序常量。

#include "Service/session/json_writer.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace mobilenet {
namespace service {
namespace {

// 追加一个 \u00XX 转义。
void AppendUnicodeEscape(U8 byte, std::string* out) {
  constexpr const C8 kHex[] = "0123456789abcdef";
  out->append("\\u00");
  out->push_back(kHex[(byte >> 4) & 0x0F]);
  out->push_back(kHex[byte & 0x0F]);
}

}  // namespace

void EscapeJsonString(const std::string& text, std::string* escaped) {
  if (escaped == nullptr) {
    return;
  }
  escaped->clear();
  escaped->reserve(text.size() + 8);
  for (const C8 ch : text) {
    const U8 byte = static_cast<U8>(ch);
    switch (byte) {
      case '"':
        escaped->append("\\\"");
        break;
      case '\\':
        escaped->append("\\\\");
        break;
      case '\b':
        escaped->append("\\b");
        break;
      case '\f':
        escaped->append("\\f");
        break;
      case '\n':
        escaped->append("\\n");
        break;
      case '\r':
        escaped->append("\\r");
        break;
      case '\t':
        escaped->append("\\t");
        break;
      default:
        if (byte < 0x20) {
          AppendUnicodeEscape(byte, escaped);
        } else {
          escaped->push_back(ch);
        }
        break;
    }
  }
}

JsonWriter::JsonWriter() = default;

void JsonWriter::BeforeValue() {
  if (failed_) {
    return;
  }
  if (levels_.empty()) {
    return;
  }
  Level& top = levels_.back();
  if (top.is_object) {
    if (expect_key_) {
      failed_ = true;
      return;
    }
    // 逗号已在 Key 中写入。
    return;
  }
  if (!top.first) {
    out_.push_back(',');
  }
  top.first = false;
}

void JsonWriter::AfterValue() {
  if (failed_) {
    return;
  }
  if (!levels_.empty() && levels_.back().is_object) {
    expect_key_ = true;
  }
}

void JsonWriter::BeginObject() {
  BeforeValue();
  if (failed_) {
    return;
  }
  out_.push_back('{');
  Level level;
  level.is_object = true;
  level.first = true;
  levels_.push_back(level);
  expect_key_ = true;
}

void JsonWriter::BeginArray() {
  BeforeValue();
  if (failed_) {
    return;
  }
  out_.push_back('[');
  Level level;
  level.is_object = false;
  level.first = true;
  levels_.push_back(level);
}

void JsonWriter::EndObject() {
  if (failed_) {
    return;
  }
  if (levels_.empty() || !levels_.back().is_object ||
      !expect_key_) {
    failed_ = true;
    return;
  }
  levels_.pop_back();
  out_.push_back('}');
  AfterValue();
}

void JsonWriter::EndArray() {
  if (failed_) {
    return;
  }
  if (levels_.empty() || levels_.back().is_object) {
    failed_ = true;
    return;
  }
  levels_.pop_back();
  out_.push_back(']');
  AfterValue();
}

void JsonWriter::Key(const std::string& key) {
  if (failed_) {
    return;
  }
  if (levels_.empty() || !levels_.back().is_object || !expect_key_) {
    failed_ = true;
    return;
  }
  Level& top = levels_.back();
  if (!top.first) {
    out_.push_back(',');
  }
  top.first = false;
  WriteString(key);
  out_.push_back(':');
  expect_key_ = false;
}

void JsonWriter::WriteString(const std::string& value) {
  std::string escaped;
  EscapeJsonString(value, &escaped);
  out_.push_back('"');
  out_.append(escaped);
  out_.push_back('"');
}

void JsonWriter::String(const std::string& value) {
  BeforeValue();
  if (failed_) {
    return;
  }
  WriteString(value);
  AfterValue();
}

void JsonWriter::Int(I64 value) {
  BeforeValue();
  if (failed_) {
    return;
  }
  out_.append(std::to_string(value));
  AfterValue();
}

void JsonWriter::Double(FP64 value) {
  BeforeValue();
  if (failed_) {
    return;
  }
  if (!std::isfinite(value)) {
    out_.append("null");
  } else {
    C8 buffer[32] = {0};
    std::snprintf(buffer, sizeof(buffer), "%.6g", value);
    out_.append(buffer);
  }
  AfterValue();
}

void JsonWriter::Boolean(Bool value) {
  BeforeValue();
  if (failed_) {
    return;
  }
  out_.append(value ? "true" : "false");
  AfterValue();
}

void JsonWriter::Null() {
  BeforeValue();
  if (failed_) {
    return;
  }
  out_.append("null");
  AfterValue();
}

void JsonWriter::Raw(const std::string& json) {
  BeforeValue();
  if (failed_) {
    return;
  }
  out_.append(json);
  AfterValue();
}

}  // namespace service
}  // namespace mobilenet
