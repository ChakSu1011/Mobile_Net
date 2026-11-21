// 模块：Runtime 最小严格 JSON 解析器实现
//
// 职责：
//   实现 RFC 8259 子集解析与 JsonValue 访问器；错误信息带字节偏移。
//
// 边界：
//   - 不实现序列化、注释、尾逗号或 JSON Pointer；
//   - 解析深度上限 256；整数溢出与浮点非有限值一律报错；
//   - 不验证未转义非 ASCII 字节的 UTF-8 合法性（manifest 由可信工具链
//     生成，且哈希校验先行）。
//
// 依赖：
//   Runtime/common/json.h、C++17 标准库。

#include "Runtime/common/json.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace runtime {
namespace {

// 允许的最大嵌套深度，防止恶意输入耗尽调用栈。
constexpr Usize kMaxJsonDepth = 256;

// 递归下降解析器。
//
// 功能：
//   在 text 上维护单一读取位置，按 JSON 语法递归构造 JsonValue。
//
// 边界：
//   - 只在 ParseJson 内部使用，不暴露；
//   - 位置只前进，不回退；错误信息基于当前 pos_。
class Parser {
 public:
  // 构造解析器。
  //
  // 参数：
  //   text：待解析文本，生命周期必须覆盖 Parser。
  explicit Parser(const std::string& text) : text_(text) {}

  // 解析整段文本。
  //
  // 参数：
  //   out：输出参数，非空。
  // 返回：
  //   成功返回 kOk；失败返回 kInvalidArgument 且 message 含偏移。
  Status Parse(JsonValue* out) {
    if (out == nullptr) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "ParseJson: output must not be null");
    }
    SkipWhitespace();
    Status status = ParseValue(out, 0);
    if (!status.ok()) {
      return status;
    }
    SkipWhitespace();
    if (pos_ != text_.size()) {
      return Error("trailing data after top-level value");
    }
    return Status::Ok();
  }

 private:
  // 构造带偏移的错误。
  //
  // 参数：
  //   detail：错误原因。
  // 返回：
  //   kInvalidArgument 状态。
  Status Error(const std::string& detail) const {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "JSON error at offset " + std::to_string(pos_) + ": " +
                          detail);
  }

  // 跳过空白字符（空格、制表、换行、回车）。
  void SkipWhitespace() {
    while (pos_ < text_.size()) {
      const C8 ch = text_[pos_];
      if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') {
        return;
      }
      ++pos_;
    }
  }

  // 当前字符是否等于 ch；匹配时前进一位。
  //
  // 参数：
  //   ch：期望字符。
  // 返回：
  //   true 表示匹配并已前进；false 表示不匹配且位置不变。
  Bool Consume(C8 ch) {
    if (pos_ < text_.size() && text_[pos_] == ch) {
      ++pos_;
      return true;
    }
    return false;
  }

  // 尝试匹配关键字（true/false/null）。
  //
  // 参数：
  //   literal：不含引号的关键字。
  // 返回：
  //   true 表示匹配并前进；false 表示不匹配且位置不变。
  Bool ConsumeLiteral(const C8* literal) {
    Usize index = 0;
    while (literal[index] != '\0') {
      if (pos_ + index >= text_.size() ||
          text_[pos_ + index] != literal[index]) {
        return false;
      }
      ++index;
    }
    pos_ += index;
    return true;
  }

  // 解析一个值。
  //
  // 参数：
  //   out：输出参数；
  //   depth：当前嵌套深度。
  // 返回：
  //   成功返回 kOk；非法输入返回 kInvalidArgument。
  Status ParseValue(JsonValue* out, Usize depth) {
    if (depth > kMaxJsonDepth) {
      return Error("nesting depth exceeds limit");
    }
    if (pos_ >= text_.size()) {
      return Error("unexpected end of input");
    }
    const C8 ch = text_[pos_];
    if (ch == '{') {
      return ParseObject(out, depth);
    }
    if (ch == '[') {
      return ParseArray(out, depth);
    }
    if (ch == '"') {
      std::string value;
      Status status = ParseString(&value);
      if (!status.ok()) {
        return status;
      }
      *out = JsonValue::MakeString(std::move(value));
      return Status::Ok();
    }
    if (ch == '-' || (ch >= '0' && ch <= '9')) {
      return ParseNumber(out);
    }
    if (ConsumeLiteral("true")) {
      *out = JsonValue::MakeBool(true);
      return Status::Ok();
    }
    if (ConsumeLiteral("false")) {
      *out = JsonValue::MakeBool(false);
      return Status::Ok();
    }
    if (ConsumeLiteral("null")) {
      *out = JsonValue::MakeNull();
      return Status::Ok();
    }
    return Error(std::string("unexpected character '") + ch + "'");
  }

  // 解析对象。
  //
  // 参数：
  //   out：输出参数；
  //   depth：当前嵌套深度。
  // 返回：
  //   成功返回 kOk；重复键、非法语法返回 kInvalidArgument。
  Status ParseObject(JsonValue* out, Usize depth) {
    Consume('{');
    std::vector<JsonMember> members;
    // 重复键检测使用哈希集合，避免大对象（如 tokenizer 词表）的 O(n²)
    // 扫描；错误语义与逐项比较一致。
    std::unordered_set<std::string> keys;
    SkipWhitespace();
    if (Consume('}')) {
      *out = JsonValue::MakeObject(std::move(members));
      return Status::Ok();
    }
    while (true) {
      SkipWhitespace();
      if (pos_ >= text_.size() || text_[pos_] != '"') {
        return Error("object key must be a string");
      }
      std::string key;
      Status status = ParseString(&key);
      if (!status.ok()) {
        return status;
      }
      if (!keys.insert(key).second) {
        return Error("duplicate object key: " + key);
      }
      SkipWhitespace();
      if (!Consume(':')) {
        return Error("expected ':' after object key");
      }
      SkipWhitespace();
      JsonValue value;
      status = ParseValue(&value, depth + 1);
      if (!status.ok()) {
        return status;
      }
      members.push_back(
          JsonMember{std::move(key), std::move(value)});
      SkipWhitespace();
      if (Consume(',')) {
        continue;
      }
      if (Consume('}')) {
        break;
      }
      return Error("expected ',' or '}' in object");
    }
    *out = JsonValue::MakeObject(std::move(members));
    return Status::Ok();
  }

  // 解析数组。
  //
  // 参数：
  //   out：输出参数；
  //   depth：当前嵌套深度。
  // 返回：
  //   成功返回 kOk；非法语法返回 kInvalidArgument。
  Status ParseArray(JsonValue* out, Usize depth) {
    Consume('[');
    std::vector<JsonValue> values;
    SkipWhitespace();
    if (Consume(']')) {
      *out = JsonValue::MakeArray(std::move(values));
      return Status::Ok();
    }
    while (true) {
      SkipWhitespace();
      JsonValue value;
      Status status = ParseValue(&value, depth + 1);
      if (!status.ok()) {
        return status;
      }
      values.push_back(std::move(value));
      SkipWhitespace();
      if (Consume(',')) {
        continue;
      }
      if (Consume(']')) {
        break;
      }
      return Error("expected ',' or ']' in array");
    }
    *out = JsonValue::MakeArray(std::move(values));
    return Status::Ok();
  }

  // 解析字符串（含转义与代理对）。
  //
  // 参数：
  //   out：输出参数，非空；写入解码后的 UTF-8 文本。
  // 返回：
  //   成功返回 kOk；非法转义/控制字符/未闭合返回 kInvalidArgument。
  Status ParseString(std::string* out) {
    Consume('"');
    while (true) {
      if (pos_ >= text_.size()) {
        return Error("unterminated string");
      }
      const U8 byte = static_cast<U8>(text_[pos_]);
      if (byte == static_cast<U8>('"')) {
        ++pos_;
        return Status::Ok();
      }
      if (byte == static_cast<U8>('\\')) {
        ++pos_;
        if (pos_ >= text_.size()) {
          return Error("unterminated escape sequence");
        }
        const C8 escape = text_[pos_++];
        switch (escape) {
          case '"':
            out->push_back('"');
            break;
          case '\\':
            out->push_back('\\');
            break;
          case '/':
            out->push_back('/');
            break;
          case 'b':
            out->push_back('\b');
            break;
          case 'f':
            out->push_back('\f');
            break;
          case 'n':
            out->push_back('\n');
            break;
          case 'r':
            out->push_back('\r');
            break;
          case 't':
            out->push_back('\t');
            break;
          case 'u': {
            U32 code_point = 0;
            Status status = ParseHex4(&code_point);
            if (!status.ok()) {
              return status;
            }
            if (code_point >= 0xd800U && code_point <= 0xdbffU) {
              // 高代理必须紧跟 \uXXXX 低代理。
              if (pos_ + 1 >= text_.size() || text_[pos_] != '\\' ||
                  text_[pos_ + 1] != 'u') {
                return Error("high surrogate without low surrogate");
              }
              pos_ += 2;
              U32 low = 0;
              status = ParseHex4(&low);
              if (!status.ok()) {
                return status;
              }
              if (low < 0xdc00U || low > 0xdfffU) {
                return Error("invalid low surrogate");
              }
              code_point = 0x10000U + ((code_point - 0xd800U) << 10) +
                           (low - 0xdc00U);
            } else if (code_point >= 0xdc00U && code_point <= 0xdfffU) {
              return Error("unexpected low surrogate");
            }
            AppendUtf8(code_point, out);
            break;
          }
          default:
            return Error("invalid escape sequence");
        }
        continue;
      }
      if (byte < 0x20U) {
        return Error("unescaped control character in string");
      }
      out->push_back(text_[pos_]);
      ++pos_;
    }
  }

  // 解析 4 位十六进制数（\\uXXXX 的后半部分）。
  //
  // 参数：
  //   value：输出参数，非空。
  // 返回：
  //   成功返回 kOk；字符不足或非十六进制返回 kInvalidArgument。
  Status ParseHex4(U32* value) {
    if (pos_ + 4 > text_.size()) {
      return Error("truncated \\u escape");
    }
    U32 result = 0;
    for (Usize index = 0; index < 4; ++index) {
      const C8 ch = text_[pos_ + index];
      U32 digit = 0;
      if (ch >= '0' && ch <= '9') {
        digit = static_cast<U32>(ch - '0');
      } else if (ch >= 'a' && ch <= 'f') {
        digit = static_cast<U32>(ch - 'a') + 10U;
      } else if (ch >= 'A' && ch <= 'F') {
        digit = static_cast<U32>(ch - 'A') + 10U;
      } else {
        return Error("invalid hexadecimal digit in \\u escape");
      }
      result = (result << 4) | digit;
    }
    pos_ += 4;
    *value = result;
    return Status::Ok();
  }

  // 把 Unicode 码点编码为 UTF-8 并追加。
  //
  // 参数：
  //   code_point：码点，取值 [0,0x10FFFF]；
  //   out：输出字符串，非空。
  void AppendUtf8(U32 code_point, std::string* out) {
    if (code_point <= 0x7fU) {
      out->push_back(static_cast<C8>(code_point));
    } else if (code_point <= 0x7ffU) {
      out->push_back(static_cast<C8>(0xc0U | (code_point >> 6)));
      out->push_back(static_cast<C8>(0x80U | (code_point & 0x3fU)));
    } else if (code_point <= 0xffffU) {
      out->push_back(static_cast<C8>(0xe0U | (code_point >> 12)));
      out->push_back(
          static_cast<C8>(0x80U | ((code_point >> 6) & 0x3fU)));
      out->push_back(static_cast<C8>(0x80U | (code_point & 0x3fU)));
    } else {
      out->push_back(static_cast<C8>(0xf0U | (code_point >> 18)));
      out->push_back(
          static_cast<C8>(0x80U | ((code_point >> 12) & 0x3fU)));
      out->push_back(
          static_cast<C8>(0x80U | ((code_point >> 6) & 0x3fU)));
      out->push_back(static_cast<C8>(0x80U | (code_point & 0x3fU)));
    }
  }

  // 解析数字；整数与浮点按语法区分。
  //
  // 参数：
  //   out：输出参数，非空。
  // 返回：
  //   成功返回 kOk；整数溢出、浮点溢出/非有限或语法错误返回
  //   kInvalidArgument。
  Status ParseNumber(JsonValue* out) {
    const Usize start = pos_;
    Consume('-');
    if (pos_ >= text_.size()) {
      return Error("truncated number");
    }
    if (text_[pos_] == '0') {
      ++pos_;
    } else if (text_[pos_] >= '1' && text_[pos_] <= '9') {
      while (pos_ < text_.size() && text_[pos_] >= '0' &&
             text_[pos_] <= '9') {
        ++pos_;
      }
    } else {
      return Error("invalid number");
    }

    Bool is_float = false;
    if (pos_ < text_.size() && text_[pos_] == '.') {
      is_float = true;
      ++pos_;
      if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') {
        return Error("expected digit after decimal point");
      }
      while (pos_ < text_.size() && text_[pos_] >= '0' &&
             text_[pos_] <= '9') {
        ++pos_;
      }
    }
    if (pos_ < text_.size() &&
        (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      is_float = true;
      ++pos_;
      if (pos_ < text_.size() &&
          (text_[pos_] == '+' || text_[pos_] == '-')) {
        ++pos_;
      }
      if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') {
        return Error("expected digit in exponent");
      }
      while (pos_ < text_.size() && text_[pos_] >= '0' &&
             text_[pos_] <= '9') {
        ++pos_;
      }
    }

    const std::string token = text_.substr(start, pos_ - start);
    const C8* token_begin = token.c_str();
    C8* token_end = nullptr;
    errno = 0;
    if (!is_float) {
      const I64 parsed = static_cast<I64>(
          std::strtoll(token_begin, &token_end, 10));
      if (errno == ERANGE ||
          token_end != token_begin + token.size()) {
        return Error("integer out of range");
      }
      *out = JsonValue::MakeInt(parsed);
      return Status::Ok();
    }
    const FP64 parsed = std::strtod(token_begin, &token_end);
    if (token_end != token_begin + token.size()) {
      return Error("invalid floating point number");
    }
    if (errno == ERANGE || !std::isfinite(parsed)) {
      return Error("floating point value out of range");
    }
    *out = JsonValue::MakeFloat(parsed);
    return Status::Ok();
  }

  // 待解析文本。
  const std::string& text_;

  // 当前读取位置。
  Usize pos_ = 0;
};

}  // namespace

JsonValue::JsonValue() = default;
JsonValue::JsonValue(const JsonValue& other) = default;
JsonValue::JsonValue(JsonValue&& other) noexcept = default;
JsonValue& JsonValue::operator=(const JsonValue& other) = default;
JsonValue& JsonValue::operator=(JsonValue&& other) noexcept = default;
JsonValue::~JsonValue() = default;

JsonValue JsonValue::MakeNull() { return JsonValue(); }

JsonValue JsonValue::MakeBool(Bool value) {
  JsonValue result;
  result.type_ = Type::kBool;
  result.bool_value_ = value;
  return result;
}

JsonValue JsonValue::MakeInt(I64 value) {
  JsonValue result;
  result.type_ = Type::kNumber;
  result.is_integer_ = true;
  result.integer_ = value;
  result.number_ = static_cast<FP64>(value);
  return result;
}

JsonValue JsonValue::MakeFloat(FP64 value) {
  JsonValue result;
  result.type_ = Type::kNumber;
  result.is_integer_ = false;
  result.number_ = value;
  return result;
}

JsonValue JsonValue::MakeString(std::string value) {
  JsonValue result;
  result.type_ = Type::kString;
  result.string_value_ = std::move(value);
  return result;
}

JsonValue JsonValue::MakeArray(std::vector<JsonValue> values) {
  JsonValue result;
  result.type_ = Type::kArray;
  result.array_value_ = std::move(values);
  return result;
}

JsonValue JsonValue::MakeObject(std::vector<JsonMember> members) {
  JsonValue result;
  result.type_ = Type::kObject;
  result.object_value_ = std::move(members);
  return result;
}

JsonValue::Type JsonValue::type() const { return type_; }

Bool JsonValue::IsNull() const { return type_ == Type::kNull; }

Bool JsonValue::IsBool() const { return type_ == Type::kBool; }

Bool JsonValue::IsNumber() const { return type_ == Type::kNumber; }

Bool JsonValue::IsString() const { return type_ == Type::kString; }

Bool JsonValue::IsArray() const { return type_ == Type::kArray; }

Bool JsonValue::IsObject() const { return type_ == Type::kObject; }

Bool JsonValue::IsInteger() const {
  return type_ == Type::kNumber && is_integer_;
}

Bool JsonValue::AsBool() const {
  return type_ == Type::kBool ? bool_value_ : false;
}

I64 JsonValue::AsInt64() const {
  if (type_ != Type::kNumber) {
    return 0;
  }
  return is_integer_ ? integer_ : static_cast<I64>(number_);
}

FP64 JsonValue::AsFloat64() const {
  if (type_ != Type::kNumber) {
    return 0.0;
  }
  return is_integer_ ? static_cast<FP64>(integer_) : number_;
}

const std::string& JsonValue::AsString() const {
  static const std::string kEmpty;
  return type_ == Type::kString ? string_value_ : kEmpty;
}

const std::vector<JsonValue>& JsonValue::AsArray() const {
  static const std::vector<JsonValue> kEmpty;
  return type_ == Type::kArray ? array_value_ : kEmpty;
}

const std::vector<JsonMember>& JsonValue::AsObject() const {
  static const std::vector<JsonMember> kEmpty;
  return type_ == Type::kObject ? object_value_ : kEmpty;
}

const JsonValue* JsonValue::Find(const std::string& key) const {
  if (type_ != Type::kObject) {
    return nullptr;
  }
  for (const JsonMember& member : object_value_) {
    if (member.key == key) {
      return &member.value;
    }
  }
  return nullptr;
}

Status ParseJson(const std::string& text, JsonValue* value) {
  Parser parser(text);
  return parser.Parse(value);
}

}  // namespace runtime
}  // namespace mobilenet
