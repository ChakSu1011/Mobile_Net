// 模块：Tokenizer ByteLevel 映射
//
// 职责：
//   实现 GPT-2 / HF ByteLevel 的 byte ↔ unicode 双向映射：把原始 UTF-8
//   字节映射为可见 Unicode 码点，供 BPE 词表使用；解码时反向还原字节。
//
// 边界：
//   - 只做字节与码点的映射，不做 BPE、归一化或特殊 token 处理；
//   - 映射表在构造后只读，可多线程共享。
//
// 依赖：
//   unicode_util.h、Runtime/common/status.h、Utils/cpp/types.h。

#ifndef MOBILENET_SERVICE_TOKENIZER_BYTE_LEVEL_H_
#define MOBILENET_SERVICE_TOKENIZER_BYTE_LEVEL_H_

#include <array>
#include <string>
#include <unordered_map>

#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace service {
namespace tokenizer_internal {

// ByteLevel 双向映射表。
//
// 功能：
//   保存 256 个字节到 Unicode 码点的映射，以及反向码点到字节的映射。
//
// 边界：
//   构造后只读；不持有任何动态资源之外的共享状态。
struct ByteLevelMap {
  // 字节 → Unicode 码点。
  std::array<I32, 256> byte_to_char{};

  // Unicode 码点 → 字节。
  std::unordered_map<I32, U8> char_to_byte;
};

// 构造 GPT-2 ByteLevel 映射。
//
// 返回：
//   映射表；可见字节映射为自身，其余按顺序映射到 U+0100 起。
ByteLevelMap MakeByteLevelMap();

// 把 UTF-8 片段编码为 ByteLevel 符号串。
//
// 参数：
//   text：合法 UTF-8 片段；
//   map：ByteLevel 映射；
//   symbols：输出参数，非空；写入每个原始字节对应的映射码点（UTF-8）。
// 返回：
//   成功返回 kOk；空指针返回 kInvalidArgument。
Status ByteLevelEncode(const std::string& text, const ByteLevelMap& map,
                      std::string* symbols);

// 把 ByteLevel 符号串还原为原始字节。
//
// 参数：
//   symbols：ByteLevel 符号串（UTF-8）；
//   map：ByteLevel 映射；
//   bytes：输出参数，非空；写入还原后的原始字节。
// 返回：
//   成功返回 kOk；符号中出现未知码点返回 kInvalidModelFile。
Status ByteLevelDecode(const std::string& symbols, const ByteLevelMap& map,
                      std::string* bytes);

}  // namespace tokenizer_internal
}  // namespace service
}  // namespace mobilenet

#endif  // MOBILENET_SERVICE_TOKENIZER_BYTE_LEVEL_H_
