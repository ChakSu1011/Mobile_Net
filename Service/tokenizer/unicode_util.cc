// 模块：Tokenizer Unicode 与 UTF-8 工具实现
//
// 职责：
//   基于 utf8proc 实现 UTF-8 解码、NFC、Unicode 类别判断与码点编码。
//
// 边界：
//   不依赖 Tokenizer 其他子模块；所有错误通过 Status 返回。
//
// 主要逻辑：
//   1. DecodeUtf8 按 utf8proc_iterate 逐码点解析并记录字节位置；
//   2. NormalizeNfc 使用 utf8proc_map 的稳定组合模式；
//   3. White_Space 按 Unicode 属性显式枚举，避免依赖类别 Zs。

#include "Service/tokenizer/unicode_util.h"

#include <cstdlib>

#include "utf8proc.h"

namespace mobilenet {
namespace service {
namespace tokenizer_internal {
namespace {

// 把 utf8proc 的错误码转换为可读原因。
const C8* Utf8procErrorName(utf8proc_ssize_t code) {
  switch (code) {
    case UTF8PROC_ERROR_NOMEM:
      return "out of memory";
    case UTF8PROC_ERROR_OVERFLOW:
      return "integer overflow";
    case UTF8PROC_ERROR_INVALIDUTF8:
      return "invalid UTF-8";
    case UTF8PROC_ERROR_NOTASSIGNED:
      return "unassigned codepoint";
    case UTF8PROC_ERROR_INVALIDOPTS:
      return "invalid options";
    default:
      return "unknown utf8proc error";
  }
}

}  // namespace

Status DecodeUtf8(const std::string& text,
                  std::vector<Codepoint>* codepoints) {
  if (codepoints == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "DecodeUtf8: codepoints must not be null");
  }
  codepoints->clear();
  codepoints->reserve(text.size());
  Usize offset = 0;
  while (offset < text.size()) {
    utf8proc_int32_t value = 0;
    const utf8proc_ssize_t length = utf8proc_iterate(
        reinterpret_cast<const utf8proc_uint8_t*>(text.data()) + offset,
        static_cast<utf8proc_ssize_t>(text.size() - offset), &value);
    if (length < 0) {
      return MakeStatus(
          ErrorCode::kInvalidArgument,
          "DecodeUtf8: invalid UTF-8 at byte " + std::to_string(offset) +
              " (" + Utf8procErrorName(length) + ")");
    }
    Codepoint item;
    item.value = static_cast<I32>(value);
    item.offset = offset;
    item.length = static_cast<Usize>(length);
    codepoints->push_back(item);
    offset += static_cast<Usize>(length);
  }
  return Status::Ok();
}

Status ValidateUtf8(const std::string& text) {
  std::vector<Codepoint> codepoints;
  return DecodeUtf8(text, &codepoints);
}

Status SanitizeUtf8(const std::string& text, std::string* sanitized) {
  if (sanitized == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "SanitizeUtf8: sanitized must not be null");
  }
  sanitized->clear();
  sanitized->reserve(text.size());
  Usize offset = 0;
  while (offset < text.size()) {
    utf8proc_int32_t value = 0;
    const utf8proc_ssize_t length = utf8proc_iterate(
        reinterpret_cast<const utf8proc_uint8_t*>(text.data()) + offset,
        static_cast<utf8proc_ssize_t>(text.size() - offset), &value);
    if (length < 0) {
      // 非法字节逐个替换为 U+FFFD，保证后续流程仍可处理。
      Status status = AppendUtf8(0xFFFD, sanitized);
      if (!status.ok()) {
        return status;
      }
      ++offset;
      continue;
    }
    sanitized->append(text.data() + offset, static_cast<Usize>(length));
    offset += static_cast<Usize>(length);
  }
  return Status::Ok();
}

Status NormalizeNfc(const std::string& text, std::string* normalized) {
  if (normalized == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "NormalizeNfc: normalized must not be null");
  }
  utf8proc_uint8_t* result = nullptr;
  const utf8proc_ssize_t length = utf8proc_map(
      reinterpret_cast<const utf8proc_uint8_t*>(text.data()),
      static_cast<utf8proc_ssize_t>(text.size()), &result,
      static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
  if (length < 0) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        std::string("NormalizeNfc: ") + Utf8procErrorName(length));
  }
  normalized->assign(reinterpret_cast<const C8*>(result),
                     static_cast<Usize>(length));
  free(result);
  return Status::Ok();
}

Bool IsLetter(I32 codepoint) {
  switch (utf8proc_category(static_cast<utf8proc_int32_t>(codepoint))) {
    case UTF8PROC_CATEGORY_LU:
    case UTF8PROC_CATEGORY_LL:
    case UTF8PROC_CATEGORY_LT:
    case UTF8PROC_CATEGORY_LM:
    case UTF8PROC_CATEGORY_LO:
      return true;
    default:
      return false;
  }
}

Bool IsNumber(I32 codepoint) {
  switch (utf8proc_category(static_cast<utf8proc_int32_t>(codepoint))) {
    case UTF8PROC_CATEGORY_ND:
    case UTF8PROC_CATEGORY_NL:
    case UTF8PROC_CATEGORY_NO:
      return true;
    default:
      return false;
  }
}

Bool IsWhitespace(I32 codepoint) {
  // Unicode White_Space 属性；与 Rust regex 的 \s 语义对齐。
  switch (codepoint) {
    case 0x0009:
    case 0x000A:
    case 0x000B:
    case 0x000C:
    case 0x000D:
    case 0x0020:
    case 0x0085:
    case 0x00A0:
    case 0x1680:
    case 0x2028:
    case 0x2029:
    case 0x202F:
    case 0x205F:
    case 0x3000:
      return true;
    default:
      break;
  }
  return codepoint >= 0x2000 && codepoint <= 0x200A;
}

Bool IsNewline(I32 codepoint) {
  return codepoint == 0x000A || codepoint == 0x000D;
}

Status AppendUtf8(I32 codepoint, std::string* out) {
  if (out == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "AppendUtf8: out must not be null");
  }
  utf8proc_uint8_t buffer[4] = {0, 0, 0, 0};
  const utf8proc_ssize_t length = utf8proc_encode_char(
      static_cast<utf8proc_int32_t>(codepoint), buffer);
  if (length <= 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "AppendUtf8: invalid codepoint " +
                          std::to_string(codepoint));
  }
  out->append(reinterpret_cast<const C8*>(buffer),
              static_cast<Usize>(length));
  return Status::Ok();
}

I32 ToLowerAscii(I32 codepoint) {
  if (codepoint >= 'A' && codepoint <= 'Z') {
    return codepoint - 'A' + 'a';
  }
  return codepoint;
}

}  // namespace tokenizer_internal
}  // namespace service
}  // namespace mobilenet
