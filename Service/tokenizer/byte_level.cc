// 模块：Tokenizer ByteLevel 映射实现
//
// 职责：
//   构造 GPT-2 ByteLevel 映射，并实现 UTF-8 字节串与符号串的互转。
//
// 边界：
//   不处理特殊 token；不验证输入是否为完整文本。
//
// 主要逻辑：
//   可见字节 '!'..'~'、'¡'..'¬'、'®'..'ÿ' 映射为自身；其余字节按
//   0..255 的顺序依次映射到 U+0100 起的码点，与 HF tokenizers 一致。

#include "Service/tokenizer/byte_level.h"

#include <vector>

#include "Service/tokenizer/unicode_util.h"

namespace mobilenet {
namespace service {
namespace tokenizer_internal {
namespace {

// 判断字节是否属于 GPT-2 可见字符集合。
Bool IsVisibleByte(U8 byte) {
  return (byte >= 0x21 && byte <= 0x7E) ||
         (byte >= 0xA1 && byte <= 0xAC) ||
         (byte >= 0xAE && byte <= 0xFF);
}

}  // namespace

ByteLevelMap MakeByteLevelMap() {
  ByteLevelMap map;
  I32 next_codepoint = 256;
  for (I32 byte = 0; byte < 256; ++byte) {
    const U8 value = static_cast<U8>(byte);
    const I32 codepoint = IsVisibleByte(value) ? byte : next_codepoint++;
    map.byte_to_char[static_cast<Usize>(byte)] = codepoint;
    map.char_to_byte.emplace(codepoint, value);
  }
  return map;
}

Status ByteLevelEncode(const std::string& text, const ByteLevelMap& map,
                      std::string* symbols) {
  if (symbols == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "ByteLevelEncode: symbols must not be null");
  }
  symbols->clear();
  symbols->reserve(text.size());
  for (const C8 ch : text) {
    const U8 byte = static_cast<U8>(ch);
    Status status =
        AppendUtf8(map.byte_to_char[static_cast<Usize>(byte)], symbols);
    if (!status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

Status ByteLevelDecode(const std::string& symbols, const ByteLevelMap& map,
                      std::string* bytes) {
  if (bytes == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "ByteLevelDecode: bytes must not be null");
  }
  bytes->clear();
  bytes->reserve(symbols.size());
  std::vector<Codepoint> codepoints;
  Status status = DecodeUtf8(symbols, &codepoints);
  if (!status.ok()) {
    return PrefixStatus(status, "ByteLevelDecode");
  }
  for (const Codepoint& item : codepoints) {
    const auto iterator = map.char_to_byte.find(item.value);
    if (iterator == map.char_to_byte.end()) {
      return MakeStatus(
          ErrorCode::kInvalidModelFile,
          "ByteLevelDecode: unknown byte-level codepoint " +
              std::to_string(item.value));
    }
    bytes->push_back(static_cast<C8>(iterator->second));
  }
  return Status::Ok();
}

}  // namespace tokenizer_internal
}  // namespace service
}  // namespace mobilenet
