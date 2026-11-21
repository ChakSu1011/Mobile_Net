// 模块：Tokenizer 资产加载与校验
//
// 职责：
//   读取并校验 tokenizer.json、tokenizer_config.json、
//   generation_config.json，构建词表、merge rank、added token、ByteLevel
//   映射与 Chat 模板支持标记。
//
// 边界：
//   - 只支持 HF BPE + ByteLevel + NFC + Qwen3 预分词正则；
//   - 不实现 truncation/padding、BOS 自动添加、特殊 token 拆分；
//   - 不执行编码；加载完成后释放 JSON 树。
//
// 依赖：
//   Runtime/common/json.h、file_util.h、byte_level.h、tokenizer.h。

#ifndef MOBILENET_SERVICE_TOKENIZER_TOKENIZER_CONFIG_H_
#define MOBILENET_SERVICE_TOKENIZER_TOKENIZER_CONFIG_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "Runtime/common/status.h"
#include "Service/tokenizer/byte_level.h"
#include "Service/tokenizer/tokenizer.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace service {
namespace tokenizer_internal {

// added token 记录。
struct AddedToken {
  // 词表 id。
  I64 id = -1;

  // 原始文本。
  std::string content;

  // 是否为 special token（解码时受 skip_special_tokens 影响）。
  Bool special = false;
};

// 加载后的 Tokenizer 资产。
struct LoadedTokenizerAssets {
  // 词表大小。
  I64 vocab_size = 0;

  // id → token 文本；空串表示该 id 无 token。
  std::vector<std::string> id_to_token;

  // token 文本 → id。
  std::unordered_map<std::string, I64> token_to_id;

  // merge 对 → rank。
  std::unordered_map<std::string, I32> merge_rank;

  // added token 列表，按 tokenizer.json 顺序。
  std::vector<AddedToken> added_tokens;

  // ByteLevel 映射。
  ByteLevelMap byte_level;

  // BOS id；未配置为 -1。
  I64 bos_token_id = -1;

  // EOS id 列表。
  std::vector<I64> eos_token_ids;

  // 是否支持 Qwen3 Chat 模板。
  Bool chat_template_supported = false;

  // 是否执行 NFC（来自 TokenizerConfig）。
  Bool enable_nfc = true;

  // 是否严格校验输入 UTF-8（来自 TokenizerConfig）。
  Bool strict_utf8 = true;
};

// 加载并校验 Tokenizer 资产。
//
// 参数：
//   config：加载配置；
//   assets：输出参数，非空；成功时写入全部资产。
// 返回：
//   成功返回 kOk；文件/解析错误返回 kIoError/kInvalidModelFile；
//   不支持的类型/字段返回 kUnsupportedOp。
Status LoadTokenizerAssets(const TokenizerConfig& config,
                           LoadedTokenizerAssets* assets);

}  // namespace tokenizer_internal
}  // namespace service
}  // namespace mobilenet

#endif  // MOBILENET_SERVICE_TOKENIZER_TOKENIZER_CONFIG_H_
