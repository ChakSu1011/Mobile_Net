// 模块：Tokenizer Unicode 与 UTF-8 工具
//
// 职责：
//   为 Tokenizer 提供 UTF-8 解码、NFC 归一化、Unicode 类别判断与码点
//   编码。Unicode 数据来自内置 utf8proc v2.10.0。
//
// 边界：
//   - 只处理 Unicode 码点与 UTF-8 字节，不涉及词表或 BPE；
//   - 不缓存、不加锁；所有函数可重入；
//   - 非法 UTF-8 通过 Status 返回，不抛出异常。
//
// 依赖：
//   utf8proc.h、Runtime/common/status.h、Utils/cpp/types.h。

#ifndef MOBILENET_SERVICE_TOKENIZER_UNICODE_UTIL_H_
#define MOBILENET_SERVICE_TOKENIZER_UNICODE_UTIL_H_

#include <string>
#include <vector>

#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace service {
namespace tokenizer_internal {

// UTF-8 文本中的一个码点及其字节位置。
//
// 说明：
//   offset/length 指向原始 UTF-8 字符串；value 为 Unicode 码点。
struct Codepoint {
  // Unicode 码点。
  I32 value = 0;

  // 在原始字符串中的字节偏移。
  Usize offset = 0;

  // UTF-8 编码字节数（1–4）。
  Usize length = 0;
};

// 解码 UTF-8 文本。
//
// 参数：
//   text：UTF-8 字节串；
//   codepoints：输出参数，非空；成功时按顺序写入全部码点。
// 返回：
//   成功返回 kOk；非法 UTF-8 返回 kInvalidArgument，消息包含字节偏移。
Status DecodeUtf8(const std::string& text,
                  std::vector<Codepoint>* codepoints);

// 校验 UTF-8 文本是否为合法序列。
//
// 参数：
//   text：待校验字节串。
// 返回：
//   合法返回 kOk；非法返回 kInvalidArgument。
Status ValidateUtf8(const std::string& text);

// 将非法 UTF-8 字节替换为 U+FFFD。
//
// 参数：
//   text：输入字节串；
//   sanitized：输出参数，非空。
// 返回：
//   始终返回 kOk；仅当 sanitized 为空指针时返回 kInvalidArgument。
Status SanitizeUtf8(const std::string& text, std::string* sanitized);

// 对 UTF-8 文本做 NFC 归一化。
//
// 参数：
//   text：合法 UTF-8 文本；
//   normalized：输出参数，非空。
// 返回：
//   成功返回 kOk；utf8proc 失败或输出为空指针返回 kInvalidArgument。
Status NormalizeNfc(const std::string& text, std::string* normalized);

// 判断码点是否为 Unicode 字母（L*）。
//
// 参数：
//   codepoint：Unicode 码点。
// 返回：
//   字母返回 true。
Bool IsLetter(I32 codepoint);

// 判断码点是否为 Unicode 数字（N*）。
//
// 参数：
//   codepoint：Unicode 码点。
// 返回：
//   数字返回 true。
Bool IsNumber(I32 codepoint);

// 判断码点是否为 Unicode White_Space。
//
// 参数：
//   codepoint：Unicode 码点。
// 返回：
//   空白返回 true。
Bool IsWhitespace(I32 codepoint);

// 判断码点是否为回车或换行。
//
// 参数：
//   codepoint：Unicode 码点。
// 返回：
//   U+000D 或 U+000A 返回 true。
Bool IsNewline(I32 codepoint);

// 将码点编码为 UTF-8 并追加到输出。
//
// 参数：
//   codepoint：Unicode 码点，必须合法；
//   out：输出参数，非空。
// 返回：
//   成功返回 kOk；非法码点或空指针返回 kInvalidArgument。
Status AppendUtf8(I32 codepoint, std::string* out);

// 把 UTF-8 文本转换为小写 ASCII（仅用于 contraction 匹配）。
//
// 参数：
//   codepoint：输入码点。
// 返回：
//   大写 ASCII 返回对应小写；其他原样返回。
I32 ToLowerAscii(I32 codepoint);

}  // namespace tokenizer_internal
}  // namespace service
}  // namespace mobilenet

#endif  // MOBILENET_SERVICE_TOKENIZER_UNICODE_UTIL_H_
