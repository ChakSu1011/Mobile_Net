// 模块：Tokenizer BPE 编码实现
//
// 职责：
//   实现预分词片段到 token ids 的 ByteLevel + BPE 流程。
//
// 边界：
//   只处理单个片段；调用方负责 NFC、预分词与 added token 分段。
//
// 主要逻辑：
//   1. ByteLevelEncode 生成符号串；
//   2. DecodeUtf8 把符号串拆成单码点符号；
//   3. 每轮找 rank 最小的相邻对，合并所有相同相邻对；
//   4. 最终符号查词表。

#include "Service/tokenizer/bpe.h"

#include <limits>

#include "Service/tokenizer/unicode_util.h"

namespace mobilenet {
namespace service {
namespace tokenizer_internal {
namespace {

// 构造 merge rank 的键；ByteLevel 符号不含 NUL，分隔安全。
std::string MakePairKey(const std::string& first, const std::string& second) {
  std::string key;
  key.reserve(first.size() + second.size() + 1);
  key.append(first);
  key.push_back('\0');
  key.append(second);
  return key;
}

}  // namespace

Status BpeEncode(const std::string& segment, const ByteLevelMap& byte_level,
                 const std::unordered_map<std::string, I64>& token_to_id,
                 const std::unordered_map<std::string, I32>& merge_rank,
                 std::vector<I64>* ids) {
  if (ids == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "BpeEncode: ids must not be null");
  }

  std::string symbol_text;
  Status status = ByteLevelEncode(segment, byte_level, &symbol_text);
  if (!status.ok()) {
    return PrefixStatus(status, "BpeEncode");
  }

  std::vector<Codepoint> codepoints;
  status = DecodeUtf8(symbol_text, &codepoints);
  if (!status.ok()) {
    return PrefixStatus(status, "BpeEncode");
  }

  std::vector<std::string> symbols;
  symbols.reserve(codepoints.size());
  for (const Codepoint& item : codepoints) {
    symbols.push_back(symbol_text.substr(item.offset, item.length));
  }

  constexpr Usize kNoIndex = static_cast<Usize>(-1);
  while (symbols.size() > 1) {
    I32 best_rank = std::numeric_limits<I32>::max();
    Usize best_index = kNoIndex;
    for (Usize index = 0; index + 1 < symbols.size(); ++index) {
      const auto iterator =
          merge_rank.find(MakePairKey(symbols[index], symbols[index + 1]));
      if (iterator != merge_rank.end() && iterator->second < best_rank) {
        best_rank = iterator->second;
        best_index = index;
      }
    }
    if (best_index == kNoIndex) {
      break;
    }

    const std::string first = symbols[best_index];
    const std::string second = symbols[best_index + 1];
    std::vector<std::string> merged;
    merged.reserve(symbols.size());
    for (Usize index = 0; index < symbols.size();) {
      if (index + 1 < symbols.size() && symbols[index] == first &&
          symbols[index + 1] == second) {
        merged.push_back(first + second);
        index += 2;
      } else {
        merged.push_back(symbols[index]);
        ++index;
      }
    }
    symbols.swap(merged);
  }

  for (const std::string& symbol : symbols) {
    const auto iterator = token_to_id.find(symbol);
    if (iterator == token_to_id.end()) {
      return MakeStatus(ErrorCode::kInternal,
                        "BpeEncode: symbol not found in vocab for segment");
    }
    ids->push_back(iterator->second);
  }
  return Status::Ok();
}

}  // namespace tokenizer_internal
}  // namespace service
}  // namespace mobilenet
